"""Dense/TF-IDF FAISS vector retriever with cosine similarity scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import settings
from src.io_utils import read_markdown
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CorpusChunk:
    """Represents an indexed text chunk from the Markdown knowledge corpus."""

    chunk_id: str
    domain: str
    filename: str
    header: str
    text: str

    @property
    def formatted_text(self) -> str:
        """Return chunk text prefixed with source metadata header."""
        return f"[{self.domain} | {self.header}] {self.text}"


class CorpusRetriever:
    """In-memory FAISS IndexFlatIP retriever over L2-normalized TF-IDF representations.

    To avoid length-mismatch attenuation between short user queries and 120-word
    paragraphs, each chunk is indexed both at the full-paragraph level and at the
    constituent-sentence level in FAISS. The cosine similarity score of a chunk
    is the maximum inner product (cosine similarity on L2-normalized vectors)
    across its paragraph vector and its constituent sentence vectors.
    """

    def __init__(self, corpus_dir: Optional[Path] = None) -> None:
        self.corpus_dir = (corpus_dir or settings.corpus_dir).resolve()
        self.chunks: List[CorpusChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._vector_to_chunk_idx: List[int] = []
        self._is_built: bool = False

    @staticmethod
    def _split_markdown_into_chunks(
        markdown_text: str, filename: str, domain: str
    ) -> List[CorpusChunk]:
        """Split a Markdown file into logical section chunks (~100-200 words)."""
        lines = markdown_text.splitlines()
        doc_title = domain.replace("_", " ").title()
        current_header = doc_title
        current_lines: List[str] = []
        chunks: List[CorpusChunk] = []
        chunk_counter = 1

        def flush_section(header: str, body_lines: List[str]) -> None:
            nonlocal chunk_counter
            body = " ".join(line.strip() for line in body_lines if line.strip()).strip()
            if not body:
                return
            chunk_id = f"{domain}-chunk-{chunk_counter:02d}"
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    domain=domain,
                    filename=filename,
                    header=header,
                    text=body,
                )
            )
            chunk_counter += 1

        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                doc_title = stripped.lstrip("# ").strip()
            elif stripped.startswith("## "):
                flush_section(current_header, current_lines)
                current_header = stripped.lstrip("# ").strip()
                current_lines = []
            else:
                current_lines.append(raw_line)

        flush_section(current_header, current_lines)
        return chunks

    @staticmethod
    def _extract_sub_passages(chunk: CorpusChunk) -> List[str]:
        """Extract full paragraph, 2-sentence windows, and individual sentences for indexing."""
        full_passage = f"{chunk.header}. {chunk.text}"
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", chunk.text)
            if len(s.strip()) > 15
        ]
        passages = [full_passage, chunk.header]
        for idx, sentence in enumerate(raw_sentences):
            passages.append(f"{chunk.header}: {sentence}")
            passages.append(sentence)
            if idx + 1 < len(raw_sentences):
                two_sent = f"{sentence} {raw_sentences[idx + 1]}"
                passages.append(two_sent)
                passages.append(f"{chunk.header}: {two_sent}")
        return passages

    def build_index(self) -> int:
        """Load all Markdown files in corpus_dir, chunk them, and build FAISS index."""
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory does not exist: {self.corpus_dir}"
            )

        md_files = sorted(self.corpus_dir.glob("*.md"))
        if not md_files:
            raise ValueError(f"No Markdown (.md) files found in {self.corpus_dir}")

        loaded_chunks: List[CorpusChunk] = []
        for md_path in md_files:
            domain = md_path.stem
            content = read_markdown(md_path)
            doc_chunks = self._split_markdown_into_chunks(
                markdown_text=content, filename=md_path.name, domain=domain
            )
            loaded_chunks.extend(doc_chunks)

        self.chunks = loaded_chunks

        index_texts: List[str] = []
        vector_to_chunk: List[int] = []
        for chunk_idx, chunk in enumerate(self.chunks):
            for passage in self._extract_sub_passages(chunk):
                index_texts.append(passage)
                vector_to_chunk.append(chunk_idx)

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9%]+\b",
        )
        tfidf_matrix = self._vectorizer.fit_transform(index_texts)
        dense_vectors = (
            normalize(tfidf_matrix, norm="l2", axis=1).toarray().astype(np.float32)
        )

        dimension = dense_vectors.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dimension)
        self._faiss_index.add(dense_vectors)
        self._vector_to_chunk_idx = vector_to_chunk
        self._is_built = True

        logger.info(
            "Built FAISS IndexFlatIP over corpus chunks",
            extra={
                "payload": {
                    "num_files": len(md_files),
                    "num_chunks": len(self.chunks),
                    "num_indexed_vectors": len(index_texts),
                    "vector_dim": dimension,
                }
            },
        )
        return len(self.chunks)

    def retrieve(
        self, query: str, top_k: int = 3, domain: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Retrieve top_k chunks and their cosine similarity scores in [0.0, 1.0]."""
        if not self._is_built or self._vectorizer is None or self._faiss_index is None:
            self.build_index()

        assert self._vectorizer is not None
        assert self._faiss_index is not None

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        query_tfidf = self._vectorizer.transform([cleaned_query])
        query_norm = float(np.linalg.norm(query_tfidf.toarray()))
        if query_norm == 0.0:
            # Query contains zero in-vocabulary tokens
            candidate_chunks = [
                c for c in self.chunks if (domain is None or c.domain == domain)
            ][:top_k]
            return [(c.formatted_text, 0.0) for c in candidate_chunks]

        query_vec = (
            normalize(query_tfidf, norm="l2", axis=1).toarray().astype(np.float32)
        )

        total_vectors = self._faiss_index.ntotal
        similarities, indices = self._faiss_index.search(query_vec, total_vectors)

        best_score_per_chunk: Dict[int, float] = {}
        for raw_score, vec_idx in zip(similarities[0], indices[0]):
            if vec_idx < 0:
                continue
            chunk_idx = self._vector_to_chunk_idx[int(vec_idx)]
            chunk_obj = self.chunks[chunk_idx]
            if domain is not None and chunk_obj.domain != domain:
                continue
            clamped_score = float(max(0.0, min(1.0, raw_score)))
            if (
                chunk_idx not in best_score_per_chunk
                or clamped_score > best_score_per_chunk[chunk_idx]
            ):
                best_score_per_chunk[chunk_idx] = clamped_score

        ranked_chunk_indices = sorted(
            best_score_per_chunk.keys(),
            key=lambda idx: best_score_per_chunk[idx],
            reverse=True,
        )[: max(1, top_k)]

        results: List[Tuple[str, float]] = [
            (self.chunks[idx].formatted_text, round(best_score_per_chunk[idx], 4))
            for idx in ranked_chunk_indices
        ]
        return results


# Module-level singleton retriever
_default_retriever = CorpusRetriever()


def retrieve(
    query: str, top_k: int = 3, domain: Optional[str] = None
) -> List[Tuple[str, float]]:
    """Retrieve top-k (chunk_text, cosine_similarity) pairs for a query."""
    return _default_retriever.retrieve(query=query, top_k=top_k, domain=domain)


def main() -> None:
    """CLI entrypoint for `make index` to build and verify the FAISS vector index."""
    num_chunks = _default_retriever.build_index()
    print(f"Indexed {num_chunks} corpus chunks across {settings.corpus_dir}.")
    sample_queries = [
        "What is the nominal operating voltage and maximum stall torque of the Dynamixel XM430-W350-T servo?",
        "What is the percentage weight of the Final Exam Prototyping Project in the ADA511 course grade?",
        "What is the adult intramuscular epinephrine dosage and concentration for acute anaphylaxis?",
    ]
    for sample_q in sample_queries:
        hits = retrieve(sample_q, top_k=2)
        top_text, top_sim = hits[0] if hits else ("NONE", 0.0)
        print(
            f"Query: {sample_q}\n  -> Top Similarity: {top_sim:.4f} | Chunk: {top_text[:95]}...\n"
        )


if __name__ == "__main__":
    main()
