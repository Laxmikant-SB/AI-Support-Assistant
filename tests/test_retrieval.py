"""
Tests for RAG Knowledge Base: Header Chunking and Hybrid Retrieval.
"""

from pathlib import Path
from src.retrieval import split_markdown_by_headers, DocumentChunk, HybridRAGPipeline


SAMPLE_MARKDOWN = """# Platform Policies

## Refund Window
Monthly subscriptions are eligible for refunds within 14 days of billing.

## Account Recovery
Users must provide verified domain email to recover locked accounts.
"""


def test_markdown_header_chunking():
    chunks = split_markdown_by_headers(SAMPLE_MARKDOWN, source_file="test_policy.md")
    assert len(chunks) == 2
    
    # Check first chunk
    assert "Refund Window" in chunks[0].header_path
    assert "14 days" in chunks[0].content
    assert chunks[0].metadata["source"] == "test_policy.md"
    
    # Check second chunk
    assert "Account Recovery" in chunks[1].header_path
    assert "locked accounts" in chunks[1].content


def test_oversized_markdown_chunking():
    long_section = "# Main Topic\n\n## Subtopic\n" + ("This is a long sentence explaining policies. " * 30)
    chunks = split_markdown_by_headers(long_section, source_file="long_doc.md", max_chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 300
