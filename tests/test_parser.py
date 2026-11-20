import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.parser import DocumentLayoutParser


def test_semantic_chunks_keep_document_intelligence_page_numbers():
    page_break = "<!-- PageBreak -->"
    first_page = "## Introduction\nFirst page text."
    second_page = "## Details\nSecond page text."
    markdown = first_page + page_break + second_page
    first_page_length = len(first_page + page_break)
    parser = DocumentLayoutParser.__new__(DocumentLayoutParser)

    chunks = list(
        parser.generate_semantic_chunks(
            markdown,
            page_spans=[
                (0, first_page_length, 1),
                (first_page_length, len(second_page), 2),
            ],
        )
    )

    assert [chunk["page_number"] for chunk in chunks] == [1, 2]
    assert all(page_break not in chunk["content"] for chunk in chunks)
