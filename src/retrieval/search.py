import os
from typing import List, Dict, Any
from azure.identity import DefaultAzureCredential
from azure.core.credentials import TokenCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    VectorFilterMode,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)


class AdvancedHybridRetriever:
    """
    Advanced multi-stage retrieval architecture executing parallel BM25 and
    Dense Vector queries, integrated with Azure AI L2 Semantic Re-scoring.
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        credential: TokenCredential | None = None,
    ):
        if not endpoint or not index_name:
            raise ValueError(
                "Search endpoint and active index name must be configured."
            )

        self.credential = credential or DefaultAzureCredential()
        self.search_client = SearchClient(
            endpoint=endpoint.rstrip("/"),
            index_name=index_name,
            credential=self.credential,
        )

    def retrieve_top_context(
        self, query_text: str, query_vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes an L2-ranked Hybrid Query to return context text segments.
        """

        # Configure the Stage 1 Dense Vector parameter
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,  # Grab wide pool before deep L2 re-ranking
            fields="content_vector",
        )

        # Execute multi-stage hybrid request
        results = self.search_client.search(
            search_text=query_text,  # BM25 Keyword Search
            vector_queries=[vector_query],  # Dense Vector Search
            query_type=QueryType.SEMANTIC,  # Activates L2 Semantic Ranker
            semantic_configuration_name="my-semantic-config",  # Must match your search index configuration
            query_caption=QueryCaptionType.EXTRACTIVE,  # Pull targeted highlights
            top=top_k,
        )

        extracted_context = []
        for result in results:
            extracted_context.append(
                {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "source_file": result.get("source_file"),
                    "page_number": result.get("page_number"),
                    "heading_hierarchy": result.get("heading_hierarchy"),
                    "search_score": result.get("@search.score"),
                    "semantic_score": result.get(
                        "@search.reranker_score"
                    ),  # L2 Cross-Encoder score
                }
            )

        return extracted_context
