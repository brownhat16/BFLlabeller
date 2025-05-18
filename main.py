"""
FastAPI Application with RAG-based Query Classification Pipeline
"""

import os
import re
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from pathlib import Path

# External dependencies
import numpy as np
import pandas as pd
from diskcache import Cache
from loguru import logger
from dotenv import load_dotenv
from rapidfuzz import process, utils
from together import Together
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Configuration Constants
CACHE_DIR = "cache"
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
DEFAULT_TOP_K = 5
FUZZY_THRESHOLD = 90

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


# -----------------------------
# Utility Functions
# -----------------------------


def clean_text(text: str) -> str:
    """Normalize text by lowercasing, cleaning whitespace, and removing punctuation."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply text cleaning to specified columns in the DataFrame."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
    return df


def build_vocab_from_column(df: pd.DataFrame, column: str, min_freq: int = 2) -> List[str]:
    """Build a vocabulary list from a column based on minimum word frequency."""
    word_counts = {}
    for val in df[column].dropna().astype(str):
        for word in val.split():
            if len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1
    return [word for word, count in word_counts.items() if count >= min_freq]


# -----------------------------
# Fuzzy Correction Logic
# -----------------------------


@lru_cache(maxsize=10_000)
def cached_fuzzy_match(token: str, vocab_key: str) -> str:
    """Cached fuzzy match using RapidFuzz, returns original token if no match found."""
    vocab = tuple(json.loads(vocab_key))
    if len(token) <= 2:
        return token
    result = process.extractOne(
        token, vocab, processor=utils.default_process, score_cutoff=FUZZY_THRESHOLD
    )
    if not result:
        return token
    match, score, _ = result
    return match if score >= FUZZY_THRESHOLD else token


def fuzzy_correct(text: str, vocab: List[str]) -> str:
    """Correct spelling in text tokens using vocabulary."""
    if not vocab:
        return text
    vocab_key = json.dumps(sorted(vocab))
    corrected = [cached_fuzzy_match(token, vocab_key) for token in text.split()]
    return " ".join(corrected)


def apply_fuzzy_correction(df: pd.DataFrame, column: str, min_freq: int = 2) -> pd.DataFrame:
    """Apply fuzzy correction to each row in a given column."""
    vocab = build_vocab_from_column(df, column, min_freq=min_freq)
    if not vocab:
        logger.warning(f"Empty vocabulary for column '{column}'. Skipping fuzzy correction.")
    df = df.copy()
    df[column] = df[column].apply(lambda x: fuzzy_correct(x, vocab))
    return df


# -----------------------------
# Embedding Retriever
# -----------------------------


class EmbeddingRetriever:
    def __init__(self, api_key: str, model_name: str = "togethercomputer/m2-bert-80M-32k-retrieval"):
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.cache = Cache(EMBEDDING_CACHE_DIR)

    def _get_cache_key(self, text: str) -> str:
        return f"{self.model_name}_{hashlib.md5(text.encode()).hexdigest()}"

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate or retrieve embedding for a single input."""
        text = text.strip()
        cache_key = self._get_cache_key(text)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        response = self.client.embeddings.create(model=self.model_name, input=text)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        if use_cache:
            self.cache[cache_key] = embedding
        return embedding

    def batch_embed(self, texts: List[str], batch_size: int = 16, use_cache: bool = True) -> np.ndarray:
        """Batch generate embeddings for multiple inputs."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            to_generate, indices, texts_to_embed = [], [], []

            if use_cache:
                for j, text in enumerate(batch):
                    text = text.strip()
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.cache:
                        while len(embeddings) <= i + j:
                            embeddings.append(None)
                        embeddings[i + j] = self.cache[cache_key]
                    else:
                        to_generate.append((j, text))
                if to_generate:
                    indices, texts_to_embed = zip(*to_generate)
            else:
                indices, texts_to_embed = zip(*[(j, text.strip()) for j, text in enumerate(batch)])

            if texts_to_embed:
                response = self.client.embeddings.create(model=self.model_name, input=list(texts_to_embed))
                for idx, j in enumerate(indices):
                    embedding = np.array(response.data[idx].embedding, dtype=np.float32)
                    if use_cache:
                        cache_key = self._get_cache_key(texts_to_embed[idx])
                        self.cache[cache_key] = embedding
                    while len(embeddings) <= i + j:
                        embeddings.append(None)
                    embeddings[i + j] = embedding

        # Fixed: Check for None values in the embedding list properly
        if any(emb is None for emb in embeddings):
            raise ValueError("Some embeddings were not generated correctly")

        # Stack embeddings after ensuring all are valid
        return np.vstack(embeddings)


# -----------------------------
# LLM Reasoner
# -----------------------------


class LLMBasedReasoner:
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.cache = {}

    def _get_cache_key(self, query: str, examples: List[Dict[str, str]]) -> str:
        examples_str = json.dumps(examples, sort_keys=True)
        return f"{query}_{hashlib.md5(examples_str.encode()).hexdigest()}"

    def classify(self, query: str, examples: List[Dict[str, str]], use_cache: bool = True) -> Dict[str, Any]:
        """Classify a query using similar labeled examples via an LLM."""
        if not examples:
            return {"label": None, "confidence": 0.0, "reasoning": "No examples provided"}
        cache_key = self._get_cache_key(query, examples)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        example_str = "\n".join([f"{i + 1}. \"{ex['query']}\" → \"{ex['label']}\"" for i, ex in enumerate(examples)])
        prompt = f"""<|begin_of_text|>
<|system|>
You are an AI assistant that helps categorize search queries into appropriate groups.
Given a user query and similar examples with their labels, determine the most suitable label for the query.
Analyze the patterns in the examples to understand the criteria for each label.
Respond in JSON format with the following fields:
- "label": The most suitable label for the query
- "confidence": A number between 0 and 1 indicating your confidence
- "reasoning": A brief explanation of your decision (1-2 sentences)
<|user|>
User query: "{query}"
Here are some similar past queries and their labels:
{example_str}
Based on the above examples, classify the user query. Return only valid JSON.
<|assistant|>
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            result.setdefault("label", examples[0]["label"])
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "No explanation provided")
            if use_cache:
                self.cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"LLM classification error: {str(e)}")
            return {
                "label": examples[0]["label"],
                "confidence": 0.3,
                "reasoning": "Fallback due to parsing error",
            }


# -----------------------------
# RAG Pipeline Core
# -----------------------------


class RAGPipeline:
    def __init__(
            self,
            df: pd.DataFrame,
            retriever: EmbeddingRetriever,
            reasoner: LLMBasedReasoner,
            query_col: str = "EP_SEARCH",
            label_col: str = "group",
            top_k: int = DEFAULT_TOP_K
    ):
        self.df = df
        self.retriever = retriever
        self.reasoner = reasoner
        self.query_col = query_col
        self.label_col = label_col
        self.top_k = top_k

        # Precompute all embeddings
        self.embeddings = self.retriever.batch_embed(df[self.query_col].fillna("").tolist())
        expected_shape = (len(df), self.embeddings.shape[1])
        if self.embeddings.shape != expected_shape:
            raise ValueError(f"Expected {expected_shape} embeddings, got {self.embeddings.shape}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-k similar examples to the given query."""
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []
        query_emb = self.retriever.embed(cleaned_query).reshape(1, -1)

        # Calculate similarities with a small epsilon to avoid division by zero
        epsilon = 1e-8
        similarities = (
                np.dot(self.embeddings, query_emb.T).flatten()
                / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + epsilon)
        )

        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        return [
            {
                "query": self.df.iloc[i][self.query_col],
                "label": self.df.iloc[i][self.label_col],
                "similarity": float(similarities[i]),
                "index": int(i),
            }
            for i in top_indices
        ]

    def predict(self, query: str) -> Dict[str, Any]:
        """Predict label for a query using retrieval-augmented generation."""
        start_time = time.time()
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return {
                "query": query,
                "cleaned_query": "",
                "predicted_label": None,
                "similar_queries": [],
                "confidence": 0.0,
                "reasoning": "Empty query after cleaning",
                "processing_time_ms": 0,
            }

        similar = self.retrieve(cleaned_query)
        if not similar:
            return {
                "query": query,
                "cleaned_query": cleaned_query,
                "predicted_label": None,
                "similar_queries": [],
                "confidence": 0.0,
                "reasoning": "No similar examples found",
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        if len({item["label"] for item in similar}) == 1:
            best = similar[0]
            return {
                "query": query,
                "cleaned_query": cleaned_query,
                "predicted_label": best["label"],
                "similar_queries": similar,
                "confidence": best["similarity"],
                "reasoning": "All retrieved examples have the same label",
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        classification = self.reasoner.classify(cleaned_query,
                                                [{"query": s["query"], "label": s["label"]} for s in similar])
        return {
            "query": query,
            "cleaned_query": cleaned_query,
            "predicted_label": classification["label"],
            "similar_queries": similar,
            "confidence": classification["confidence"],
            "reasoning": classification["reasoning"],
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }


# -----------------------------
# API Models
# -----------------------------


class QueryInput(BaseModel):
    query: str = Field(..., description="Search query to classify")

    @validator("query")
    def query_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class PredictionOutput(BaseModel):
    query: str
    cleaned_query: str
    predicted_label: Optional[str]
    similar_queries: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    processing_time_ms: int


class BulkQueryInput(BaseModel):
    queries: List[str] = Field(..., description="List of queries to classify")

    @validator("queries")
    def queries_not_empty(cls, v):
        if not v:
            raise ValueError("Queries list cannot be empty")
        return [q.strip() for q in v if q.strip()]


# -----------------------------
# FastAPI App Setup
# -----------------------------


app = FastAPI(
    title="RAG Query Labeler API",
    description="API for classifying search queries using Retrieval-Augmented Generation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
df = None
pipeline = None


@app.on_event("startup")
async def startup_event():
    global df, pipeline
    try:
        logger.info("Starting up the application...")
        data_path = os.getenv("DATA_PATH", "labelled.csv")
        df = pd.read_csv(data_path)
        df = clean_dataframe(df, columns=["EP_SEARCH", "group"])
        df = apply_fuzzy_correction(df, "EP_SEARCH", min_freq=2)

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set in environment variables.")

        retriever = EmbeddingRetriever(api_key=api_key)
        reasoner = LLMBasedReasoner(api_key=api_key)
        pipeline = RAGPipeline(df, retriever, reasoner)
        logger.info("Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
async def predict_label(query_input: QueryInput):
    try:
        result = pipeline.predict(query_input.query)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_bulk", response_model=List[PredictionOutput])
async def predict_bulk(bulk_input: BulkQueryInput):
    try:
        return [pipeline.predict(q) for q in bulk_input.queries]
    except Exception as e:
        logger.error(f"Bulk prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
