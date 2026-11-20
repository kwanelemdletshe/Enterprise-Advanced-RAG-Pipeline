import os
import argparse
import uuid
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

# Local module imports
from ingestion.parser import DocumentLayoutParser
from inference import embed_texts


def run_indexing_pipeline(data_dir: str):
    """
    Scans a local repository directory, runs markdown parsing, generates dense
    vector embeddings using text-embedding-3-large, and commits chunks to the search index.
    """
    load_dotenv()

    # Initialize connection environment values
    project_endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
    document_endpoint = os.getenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", project_endpoint
    )
    inference_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    embedding_model = os.getenv(
        "AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"
    )
    embedding_dimensions = int(
        os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1536")
    )

    if not project_endpoint:
        raise ValueError(
            "CRITICAL: AZURE_PROJECT_ENDPOINT is missing or empty in your configuration setup."
        )
    if not search_endpoint:
        raise ValueError("AZURE_SEARCH_ENDPOINT must be configured.")
    if not index_name:
        raise ValueError("AZURE_SEARCH_INDEX_NAME must be configured.")
    if not document_endpoint:
        raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT must be configured.")

    print("[*] Instantiating Azure Entra ID keyless connections...")

    # Modern context architecture for Azure AI Foundry
    with DefaultAzureCredential() as credential:
        print("[*] Authenticating with Microsoft Foundry via Project Endpoint...")

        # Correct keyword parameter schema for 2.x releases
        if not inference_endpoint:
            raise ValueError(
                "AZURE_INFERENCE_ENDPOINT must be configured."
            )

        # Target Search Index Client
        search_client = SearchClient(
            endpoint=search_endpoint, index_name=index_name, credential=credential
        )

        # Document Parsing layer
        parser = DocumentLayoutParser(endpoint=document_endpoint)

        if not os.path.exists(data_dir):
            print(f"[!] Target directory path {data_dir} does not exist.")
            return

        print(f"[*] Commencing parsing operation on data folder: {data_dir}")
        upload_batch = []

        for filename in os.listdir(data_dir):
            if filename.lower().endswith((".pdf", ".docx", ".txt")):
                file_path = os.path.join(data_dir, filename)
                print(f"[+] Processing layout extraction for: {filename}")

                try:
                    parsed_document = parser.parse_document(file_path)
                    raw_markdown = parsed_document.content
                    if raw_markdown is None:
                        raise ValueError(
                            f"Document Intelligence returned no content for {filename}."
                        )

                    page_spans = []
                    for page in parsed_document.pages or []:
                        spans = page.spans or []
                        if spans:
                            start = min(span.offset for span in spans)
                            end = max(span.offset + span.length for span in spans)
                            page_spans.append(
                                (start, end - start, page.page_number)
                            )
                    if not page_spans:
                        raise ValueError(
                            f"Document Intelligence returned no page spans for {filename}."
                        )

                    chunks = list(
                        parser.generate_semantic_chunks(
                            raw_markdown,
                            max_chunk_tokens=512,
                            page_spans=page_spans,
                        )
                    )

                    file_documents = []
                    for idx, chunk in enumerate(chunks):
                        vectors = embed_texts(
                            credential=credential,
                            model=embedding_model,
                            texts=[chunk["content"]],
                            dimensions=embedding_dimensions,
                        )
                        vector_data = vectors[0]

                        search_document = {
                            "id": str(
                                uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{idx}")
                            ),
                            "content": chunk["content"],
                            "content_vector": vector_data,
                            "source_file": filename,
                            "page_number": chunk["page_number"],
                            "heading_hierarchy": chunk["heading_hierarchy"],
                        }
                        file_documents.append(search_document)
                    upload_batch.extend(file_documents)

                except Exception as e:
                    print(
                        f"[!] Failed parsing pipeline execution for file {filename}: {str(e)}"
                    )
                    continue

        if upload_batch:
            print(
                f"[*] Committing bulk batch of {len(upload_batch)} chunks to search index: {index_name}..."
            )
            results = search_client.upload_documents(documents=upload_batch)
            failed_results = [result for result in results if not result.succeeded]
            if failed_results:
                raise RuntimeError(
                    f"Failed to index {len(failed_results)} document chunks."
                )

            for filename in {
                document["source_file"] for document in upload_batch
            }:
                escaped_filename = filename.replace("'", "''")
                existing_documents = search_client.search(
                    search_text="*",
                    filter=f"source_file eq '{escaped_filename}'",
                    select=["id"],
                    top=1000,
                )
                current_ids = {
                    document["id"]
                    for document in upload_batch
                    if document["source_file"] == filename
                }
                stale_ids = [
                    result["id"]
                    for result in existing_documents
                    if result["id"] not in current_ids
                ]
                if stale_ids:
                    delete_results = search_client.delete_documents(
                        documents=[{"id": document_id} for document_id in stale_ids]
                    )
                    if any(not result.succeeded for result in delete_results):
                        raise RuntimeError(
                            f"Failed to remove stale chunks for {filename}."
                        )

            print(f"[OK] Successfully indexed {len(results)} layout elements.")
        else:
            print(
                "[-] No valid content data extractions were staged for indexing operations."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced RAG Ingestion Driver Utilities"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory container holding target ingestion file documents.",
    )
    args = parser.parse_args()

    run_indexing_pipeline(args.data_dir)
