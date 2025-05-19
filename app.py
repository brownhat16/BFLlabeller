import streamlit as st
import requests
import json

# Define the API base URL
API_BASE_URL = "https://bfllabeller.onrender.com "  # Replace with your actual API URL

# Function to perform health check
def check_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return "API is healthy!"
        else:
            return f"Health check failed with status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to predict a single label
def predict_single(query):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"query": query}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return f"Prediction failed with status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to predict bulk labels
def predict_bulk(queries):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_bulk",
            json={"queries": queries}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return f"Bulk prediction failed with status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Main Streamlit app
def main():
    st.title("RAG Query Labeler API Demo")

    # Sidebar for navigation
    menu = st.sidebar.selectbox("Menu", ["Health Check", "Predict Single Query", "Predict Bulk Queries"])

    if menu == "Health Check":
        st.header("Health Check")
        if st.button("Check API Health"):
            result = check_health()
            st.write(result)

    elif menu == "Predict Single Query":
        st.header("Predict Single Query")
        query = st.text_input("Enter your query:")
        if st.button("Predict"):
            if query:
                result = predict_single(query)
                st.write("Prediction Result:")
                st.json(result)
            else:
                st.warning("Please enter a query.")

    elif menu == "Predict Bulk Queries":
        st.header("Predict Bulk Queries")
        queries_input = st.text_area("Enter queries (one per line):")
        if st.button("Predict Bulk"):
            if queries_input:
                queries = [q.strip() for q in queries_input.split("\n") if q.strip()]
                result = predict_bulk(queries)
                st.write("Bulk Prediction Result:")
                st.json(result)
            else:
                st.warning("Please enter at least one query.")

if __name__ == "__main__":
    main()
