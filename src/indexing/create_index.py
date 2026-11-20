import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)


def create_advanced_search_index():
    load_dotenv()

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    embedding_dimensions = int(
        os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1536")
    )

    if not search_endpoint or not index_name:
        raise ValueError(
            "Ensure AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX_NAME are configured in your env file."
        )
    search_endpoint = search_endpoint.rstrip("/")

    print(
        f"[*] Initializing connection to Azure AI Search Service at: {search_endpoint}"
    )
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    # 1. Define Fields Schema (Matching our Ingestion Pipeline output)
    fields = [
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True, filterable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="my-vector-profile",
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="page_number", type=SearchFieldDataType.Int32, filterable=True
        ),
        SearchableField(
            name="heading_hierarchy", type=SearchFieldDataType.String, filterable=True
        ),
    ]

    # 2. Configure Vector Search Vectorization Parameters (HNSW)
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile", algorithm_configuration_name="my-hnsw-config"
            )
        ],
    )

    # 3. Configure the L2 Deep Semantic Ranker Profile
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=None,
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="heading_hierarchy")],
                ),
            )
        ]
    )

    # 4. Construct complete index metadata profile definition
    index_definition = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    print(f"[*] Provisioning search index schema named '{index_name}' in the cloud...")
    index_client.create_or_update_index(index_definition)
    print("[OK] Index created successfully and ready for data ingestion vectors.")


if __name__ == "__main__":
    create_advanced_search_index()
