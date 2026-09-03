"""
Stage 3: RAG Knowledge Base Retrieval Pipeline
- Structure-aware Markdown header chunking
- Dense retrieval with sentence-transformers ('all-MiniLM-L6-v2') + ChromaDB
- Sparse retrieval with BM25 (rank_bm25)
- Hybrid search via Reciprocal Rank Fusion (RRF)
- Re-ranking with CrossEncoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    id: str
    content: str
    source_file: str
    header_path: str
    metadata: Dict[str, Any]


def split_markdown_by_headers(
    text: str,
    source_file: str,
    max_chunk_size: int = 600,
    chunk_overlap: int = 60
) -> List[DocumentChunk]:
    """
    Structure-aware chunking:
    1. Splits Markdown documents by headers (H1, H2, H3).
    2. Attaches header path hierarchy to metadata for context.
    3. Falls back to character splitting with overlap for oversized sections.
    """
    lines = text.split("\n")
    chunks: List[DocumentChunk] = []
    
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_section_lines = []
    chunk_idx = 0

    def flush_section():
        nonlocal chunk_idx, current_section_lines
        content = "\n".join(current_section_lines).strip()
        if not content:
            return
        
        header_path = " > ".join(filter(None, [current_h1, current_h2, current_h3])) or "Overview"
        
        # If section is small enough, keep as single chunk
        if len(content) <= max_chunk_size:
            chunks.append(DocumentChunk(
                id=f"{Path(source_file).stem}_{chunk_idx}",
                content=f"[{header_path}]\n{content}",
                source_file=source_file,
                header_path=header_path,
                metadata={"source": source_file, "header": header_path}
            ))
            chunk_idx += 1
        else:
            # Recursive sub-chunking for oversized sections
            start = 0
            while start < len(content):
                end = min(start + max_chunk_size, len(content))
                # Try not to split in the middle of a sentence
                if end < len(content):
                    last_period = content.rfind(". ", start, end)
                    if last_period != -1 and last_period > start + (max_chunk_size // 2):
                        end = last_period + 1
                
                sub_text = content[start:end].strip()
                if sub_text:
                    chunks.append(DocumentChunk(
                        id=f"{Path(source_file).stem}_{chunk_idx}",
                        content=f"[{header_path}]\n{sub_text}",
                        source_file=source_file,
                        header_path=header_path,
                        metadata={"source": source_file, "header": header_path}
                    ))
                    chunk_idx += 1
                start = end - chunk_overlap if end < len(content) else len(content)

        current_section_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            flush_section()
            current_h1 = stripped[2:].strip()
            current_h2 = ""
            current_h3 = ""
        elif stripped.startswith("## "):
            flush_section()
            current_h2 = stripped[3:].strip()
            current_h3 = ""
        elif stripped.startswith("### "):
            flush_section()
            current_h3 = stripped[4:].strip()
        else:
            current_section_lines.append(line)

    flush_section()
    return chunks


class HybridRAGPipeline:
    """Hybrid RAG pipeline combining Chroma dense search, BM25, and Cross-Encoder re-ranking."""

    def __init__(
        self,
        docs_dir: Path,
        persist_dir: Optional[Path] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = persist_dir or (self.docs_dir.parent / "data" / "chroma_db")
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        
        self.chunks: List[DocumentChunk] = []
        self.bm25 = None
        self.bm25_corpus = []
        self.chroma_collection = None
        self.reranker = None
        
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads documents, chunks them, builds BM25 index and Chroma vectorstore."""
        print(f"Loading and chunking Markdown files from {self.docs_dir}...")
        self.chunks = []
        for md_file in sorted(self.docs_dir.glob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                file_text = f.read()
            doc_chunks = split_markdown_by_headers(file_text, source_file=md_file.name)
            self.chunks.extend(doc_chunks)
            
        print(f"Created {len(self.chunks)} structured chunks from {len(list(self.docs_dir.glob('*.md')))} documents.")

        # 1. Initialize BM25
        from rank_bm25 import BM25Okapi
        self.bm25_corpus = [c.content.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)

        # 2. Initialize Chroma + SentenceTransformer
        import chromadb
        from chromadb.utils import embedding_functions

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        self.chroma_collection = chroma_client.get_or_create_collection(
            name="support_kb",
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Populate Chroma if empty
        if self.chroma_collection.count() == 0:
            print("Populating Chroma vector database...")
            self.chroma_collection.add(
                ids=[c.id for c in self.chunks],
                documents=[c.content for c in self.chunks],
                metadatas=[c.metadata for c in self.chunks]
            )
            print(f"Indexed {self.chroma_collection.count()} chunks in Chroma.")

    def _get_reranker(self):
        """Lazy load CrossEncoder reranker to conserve initial memory."""
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading CrossEncoder: {self.reranker_model_name}...")
            self.reranker = CrossEncoder(self.reranker_model_name)
        return self.reranker

    def search_dense(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Dense similarity search via Chroma."""
        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.chunks))
        )
        hits = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "dense_distance": results["distances"][0][i] if "distances" in results else 0.0,
                    "dense_rank": i + 1
                })
        return hits

    def search_sparse(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Sparse keyword search via BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        hits = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            hits.append({
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "bm25_score": float(scores[idx]),
                "sparse_rank": rank + 1
            })
        return hits

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) combining Dense and Sparse BM25 ranks.
        RRF Score(d) = 1 / (rrf_k + dense_rank) + 1 / (rrf_k + sparse_rank)
        """
        dense_results = self.search_dense(query, top_k=top_k * 2)
        sparse_results = self.search_sparse(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}

        for item in dense_results:
            doc_id = item["id"]
            doc_lookup[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + item["dense_rank"]))

        for item in sparse_results:
            doc_id = item["id"]
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + item["sparse_rank"]))

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]
        
        hybrid_hits = []
        for doc_id in sorted_ids:
            hit = dict(doc_lookup[doc_id])
            hit["rrf_score"] = round(rrf_scores[doc_id], 6)
            hybrid_hits.append(hit)
            
        return hybrid_hits

    def retrieve_and_rerank(self, query: str, top_k: int = 3, candidate_k: int = 10) -> List[Dict[str, Any]]:
        """Full retrieval pipeline: Hybrid Search (Candidate K) -> Cross-Encoder Re-ranking -> Top K."""
        candidates = self.hybrid_search(query, top_k=candidate_k)
        if not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [[query, c["content"]] for c in candidates]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by cross-encoder score descending
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
        return ranked
