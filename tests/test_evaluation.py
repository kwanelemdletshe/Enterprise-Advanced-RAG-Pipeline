import os
import sys
import pytest
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
)
load_dotenv()

# Local driver components for test injection
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval.search import AdvancedHybridRetriever
from generation.orchestrator import RAGGenerationOrchestrator

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("AZURE_SEARCH_ENDPOINT")
        and os.getenv("AZURE_SEARCH_INDEX_NAME")
        and (
            (
                os.getenv("AZURE_PROJECT_CONNECTION_STRING")
                and "<" not in os.getenv("AZURE_PROJECT_CONNECTION_STRING")
            )
            or (
                os.getenv("AZURE_PROJECT_ENDPOINT")
                and "<" not in os.getenv("AZURE_PROJECT_ENDPOINT")
                and os.getenv("AZURE_SUBSCRIPTION_ID")
                and os.getenv("AZURE_RESOURCE_GROUP")
                and os.getenv("AZURE_PROJECT_NAME")
            )
        )
    ),
    reason="Live Azure RAG evaluation environment is not configured",
)


@pytest.fixture(scope="module")
def setup_rag_pipeline():
    """
    Sets up pipeline resources using centralized corporate environment variables.
    """
    load_dotenv()

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    llm_model = os.getenv("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4o")
    live_configured = (
        search_endpoint
        and index_name
        and "<" not in search_endpoint
        and "<" not in index_name
        and (
            (
                os.getenv("AZURE_PROJECT_CONNECTION_STRING")
                and "<" not in os.getenv("AZURE_PROJECT_CONNECTION_STRING")
            )
            or (
                os.getenv("AZURE_PROJECT_ENDPOINT")
                and "<" not in os.getenv("AZURE_PROJECT_ENDPOINT")
                and os.getenv("AZURE_SUBSCRIPTION_ID")
                and os.getenv("AZURE_RESOURCE_GROUP")
                and os.getenv("AZURE_PROJECT_NAME")
            )
        )
    )
    if not live_configured:
        pytest.skip("Live Azure RAG evaluation environment is not configured")

    credential = DefaultAzureCredential()

    retriever = AdvancedHybridRetriever(endpoint=search_endpoint, index_name=index_name)
    try:
        orchestrator = RAGGenerationOrchestrator(
            credential=credential, model_deployment=llm_model
        )
    except ValueError as exc:
        pytest.skip(f"Live Azure RAG evaluation environment is not configured: {exc}")

    try:
        index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=credential,
        )
        index_client.get_service_statistics()
    except AzureError as exc:
        pytest.skip(
            f"Live Azure Search service is unavailable: {exc}"
        )

    return llm_model, retriever, orchestrator


def test_rag_triad_quality_metrics(setup_rag_pipeline):
    """
    Executes deep evaluation checking Groundedness and Context Relevance
    using the formal Azure AI Evaluation SDK validation layers.
    """
    llm_model, retriever, orchestrator = setup_rag_pipeline

    # Golden evaluation test set dataset simulating enterprise user entries
    test_query = "What is the security compliance policy for remote database mutations?"
    embedding_dimensions = int(
        os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1536")
    )
    mock_query_vector = [
        0.01
    ] * embedding_dimensions

    # 1. Fetch retrieval context payload
    context_segments = retriever.retrieve_top_context(
        query_text=test_query, query_vector=mock_query_vector, top_k=2
    )

    # 2. Extract raw context text string sequence block
    flattened_context = " ".join([doc["content"] for doc in context_segments])

    # 3. Generate engine response sequence
    generated_response = orchestrator.generate_response(
        query=test_query, context_segments=context_segments
    )

    # 4. Instantiate Evaluation metrics evaluators targeting our AI Foundry Workspace setup
    model_config = {
        "azure_endpoint": os.getenv("AZURE_PROJECT_ENDPOINT"),
        "azure_deployment": llm_model,
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    }
    groundedness_eval = GroundednessEvaluator(model_config=model_config)
    relevance_eval = RelevanceEvaluator(model_config=model_config)

    # 5. Programmatically evaluate execution metrics tracking loops
    groundedness_score = groundedness_eval(
        response=generated_response, context=flattened_context
    )

    relevance_score = relevance_eval(query=test_query, response=generated_response)

    print(
        f"\n[EVALUATION METRICS] Groundedness Score: {groundedness_score.get('groundedness')}"
    )
    print(f"\n[EVALUATION METRICS] Relevance Score: {relevance_score.get('relevance')}")

    # Enterprise Grade Guardrail Threshold Test Assertions
    assert (
        groundedness_score.get("groundedness") >= 4.0
    ), "Pipeline failed grounding checks. Model output exhibits loose contextual adherence or hallucinations."
    assert (
        relevance_score.get("relevance") >= 4.0
    ), "Pipeline failed relevance checks. Response drift detected away from prompt parameters."
