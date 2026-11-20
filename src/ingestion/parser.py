import os
from typing import Any, Dict, Generator, Sequence
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Cleaned import: Remove AnalyzeParagraph
from azure.ai.documentintelligence.models import AnalyzeResult

class DocumentLayoutParser:
    """
    Enterprise Document Parser utilizing Azure AI Document Intelligence
    to extract layout structures and preserve high-fidelity Markdown formats.
    """

    def __init__(self, endpoint: str):
        if not endpoint:
            raise ValueError(
                "Azure AI Document Intelligence endpoint must be provided."
            )

        # Keyless authentication using enterprise Entra ID
        self.credential = DefaultAzureCredential()
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=self.credential
        )

    def parse_document(self, file_path: str) -> AnalyzeResult:
        """
        Parses a document and returns its content plus page-offset metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")

        with open(file_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                output_content_format="markdown",
                content_type="application/octet-stream",
            )

        return poller.result()

    def parse_to_markdown(self, file_path: str) -> str:
        """Parses a document into GitHub-Flavored Markdown."""
        result = self.parse_document(file_path)
        if result.content is None:
            raise ValueError(f"No Markdown content was returned for: {file_path}")
        return result.content

    def generate_semantic_chunks(
        self,
        markdown_content: str,
        max_chunk_tokens: int = 512,
        page_spans: Sequence[tuple[int, int, int | None]] | None = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Splits markdown text on structural boundaries (like sections and headers)
        instead of naive, destructive character cuts, keeping chunks on their source page.
        """
        if page_spans is None:
            page_spans = [(0, len(markdown_content), None)]
        if not page_spans:
            raise ValueError("Document page spans are required to generate chunks.")

        current_chunk = []
        current_heading = "Root"
        current_token_estimate = 0

        def make_chunk(page_number: int | None) -> Dict[str, Any]:
            chunk = {
                "content": "\n".join(current_chunk),
                "heading_hierarchy": current_heading,
            }
            if page_number is not None:
                chunk["page_number"] = page_number
            return chunk

        for offset, length, page_number in page_spans:
            page_content = markdown_content[offset : offset + length].replace(
                "<!-- PageBreak -->", ""
            )
            for line in page_content.split("\n"):
                if line.startswith("#"):
                    if current_chunk:
                        yield make_chunk(page_number)
                        current_chunk = []
                        current_token_estimate = 0
                    current_heading = line.strip("# ")

                line_token_estimate = len(line) // 4
                if (
                    current_token_estimate + line_token_estimate > max_chunk_tokens
                    and current_chunk
                ):
                    yield make_chunk(page_number)
                    current_chunk = [line]
                    current_token_estimate = line_token_estimate
                else:
                    current_chunk.append(line)
                    current_token_estimate += line_token_estimate

            if current_chunk:
                yield make_chunk(page_number)
                current_chunk = []
                current_token_estimate = 0
