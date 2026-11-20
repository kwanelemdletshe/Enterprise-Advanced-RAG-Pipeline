import os
import argparse
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
# Local module imports
from retrieval.search import AdvancedHybridRetriever
from generation.orchestrator import RAGGenerationOrchestrator
from inference import embed_texts


def run_query_pipeline(user_query: str):
    """
    Orchestrates the entire advanced RAG process: converts query to vector,
    performs multi-stage hybrid search, enforces guardrails, and renders responses with inline citations.
    """
    load_dotenv()

    # Read active deployment infrastructure coordinates
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    inference_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    llm_model = os.getenv("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4o")
    embedding_model = os.getenv(
        "AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"
    )
    embedding_dimensions = int(
        os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1536")
    )

    if not search_endpoint:
        raise ValueError("AZURE_SEARCH_ENDPOINT must be configured.")
    if not inference_endpoint:
        raise ValueError("AZURE_INFERENCE_ENDPOINT must be configured.")
    if not index_name:
        raise ValueError("AZURE_SEARCH_INDEX_NAME must be configured.")

    print(f"[*] Processing incoming user prompt: '{user_query}'")
    credential = DefaultAzureCredential()

    # 1. Generate text embedding matrix via unified client abstractions
    print("[*] Generating dense vector embeddings...")
    query_vector = embed_texts(
        credential=credential,
        model=embedding_model,
        texts=[user_query],
        dimensions=embedding_dimensions,
    )[0]

    # 2. Stage 1/2 Multi-Stage Context Retrieval (BM25 + Dense Vector + L2 Semantic Ranking)
    print("[*] Executing Hybrid Vector search with L2 Cross-Encoder re-scoring...")
    retriever = AdvancedHybridRetriever(endpoint=search_endpoint, index_name=index_name)
    context_segments = retriever.retrieve_top_context(
        query_text=user_query, query_vector=query_vector, top_k=5
    )

    print(
        f"[OK] Retrieved {len(context_segments)} context fragments from database index."
    )
    for doc in context_segments:
        print(
            f"    - Score: {doc['search_score']:.4f} | Rerank Score: {doc['semantic_score']:.4f} | Source: {doc['source_file']} (Page {doc['page_number']})"
        )

    # 3. Grounded Generation and Guardrail Validation Evaluation
    print("[*] Generating context-grounded response via Azure OpenAI...")
    orchestrator = RAGGenerationOrchestrator(
        credential=credential, model_deployment=llm_model
    )

    final_output = orchestrator.generate_response(
        query=user_query, context_segments=context_segments
    )

    print("\n" + "=" * 50 + "\n SYSTEM GENERATED SYSTEM RESPONSE \n" + "=" * 50)
    print(final_output)
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced RAG Query Runtime Engine")
    parser.add_argument(
        "--query",
        type=str,
        help="The question or search query to pass into the enterprise data repository pipeline.",
    )
    args = parser.parse_args()
    user_query = args.query

    if user_query is None:
        try:
            user_query = input("Enter your question: ")
        except EOFError:
            parser.error("No question was entered.")

    user_query = user_query.strip()
    if not user_query:
        parser.error("Question cannot be empty.")

    run_query_pipeline(user_query)


if __name__ == "__main__":
    main()
