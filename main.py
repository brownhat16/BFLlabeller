"""
FastAPI Application with RAG-based Query Classification Pipeline and Enhanced Word Completion
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
from together import Together
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Import the enhanced fuzzy matching implementation
from main import (
    build_enhanced_vocab,
    enhanced_text_correction,
    apply_enhanced_correction
)

# Load environment variables
load_dotenv()

# Configuration Constants
CACHE_DIR = "cache"
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
DEFAULT_TOP_K = 5
MIN_WORD_FREQ = 2

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
            top_k: int = DEFAULT_TOP_K,
            vocab: Optional[Dict[str, int]] = None
    ):
        self.df = df
        self.retriever = retriever
        self.reasoner = reasoner
        self.query_col = query_col
        self.label_col = label_col
        self.top_k = top_k
        
        # Store vocabulary for text correction
        self.vocab = vocab or build_enhanced_vocab(df, query_col, min_freq=MIN_WORD_FREQ)

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
        
        # Apply enhanced correction with word completion
        corrected_result = enhanced_text_correction(cleaned_query, self.vocab, include_metadata=True)
        if isinstance(corrected_result, dict):
            corrected_query = corrected_result["corrected_text"]
            corrections = corrected_result["corrections"]
        else:
            corrected_query = corrected_result
            corrections = []
            
        if not corrected_query:
            return {
                "query": query,
                "cleaned_query": "",
                "corrected_query": "",
                "word_corrections": [],
                "predicted_label": None,
                "similar_queries": [],
                "confidence": 0.0,
                "reasoning": "Empty query after cleaning",
                "processing_time_ms": 0,
            }

        similar = self.retrieve(corrected_query)
        if not similar:
            return {
                "query": query,
                "cleaned_query": cleaned_query,
                "corrected_query": corrected_query,
                "word_corrections": corrections,
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
                "corrected_query": corrected_query,
                "word_corrections": corrections,
                "predicted_label": best["label"],
                "similar_queries": similar,
                "confidence": best["similarity"],
                "reasoning": "All retrieved examples have the same label",
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        classification = self.reasoner.classify(corrected_query,
                                              [{"query": s["query"], "label": s["label"]} for s in similar])
        return {
            "query": query,
            "cleaned_query": cleaned_query,
            "corrected_query": corrected_query,
            "word_corrections": corrections,
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
    corrected_query: str
    word_corrections: List[Dict[str, Any]]
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
    description="API for classifying search queries using Retrieval-Augmented Generation with enhanced word completion",
    version="2.1.0",
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
vocabulary = None


@app.on_event("startup")
async def startup_event():
    global df, pipeline, vocabulary
    try:
        logger.info("Starting up the application...")
        data_path = os.getenv("DATA_PATH", "labelled.csv")
        df = pd.read_csv(data_path)
        df = clean_dataframe(df, columns=["EP_SEARCH", "group"])
        
        # Build vocabulary from the cleaned data
        vocabulary = build_enhanced_vocab(df, "EP_SEARCH", min_freq=MIN_WORD_FREQ)
        logger.info(f"Built vocabulary with {len(vocabulary)} terms")
        
        # Apply the enhanced correction with word completion
        df = apply_enhanced_correction(df, "EP_SEARCH", min_freq=MIN_WORD_FREQ)
        logger.info("Applied enhanced text correction")

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set in environment variables.")

        retriever = EmbeddingRetriever(api_key=api_key)
        reasoner = LLMBasedReasoner(api_key=api_key)
        pipeline = RAGPipeline(df, retriever, reasoner, vocab=vocabulary)
        logger.info("Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    return {"status": "healthy", "vocab_size": len(vocabulary) if vocabulary else 0}


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


# Endpoint to view vocabulary and test word completion
@app.get("/vocabulary", response_model=Dict[str, Any])
async def get_vocabulary_stats():
    if not vocabulary:
        raise HTTPException(status_code=500, detail="Vocabulary not initialized")
    
    # Get top words by frequency
    top_words = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:100]
    
    return {
        "vocabulary_size": len(vocabulary),
        "top_words": dict(top_words),
    }


@app.post("/test_completion")
async def test_word_completion(query_input: QueryInput):
    if not vocabulary:
        raise HTTPException(status_code=500, detail="Vocabulary not initialized")
    
    original = query_input.query
    cleaned = clean_text(original)
    
    # Test the completion algorithm
    result = enhanced_text_correction(cleaned, vocabulary, include_metadata=True)
    
    return {
        "original": original,
        "cleaned": cleaned,
        "result": result
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
