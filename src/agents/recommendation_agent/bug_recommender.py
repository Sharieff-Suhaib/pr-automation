"""Similar Bug Recommender.

Answers: *which historical bugs resemble this issue, and how were they fixed?*

Given a new issue, retrieves the top-k most similar bugs from a small JSON
corpus (data/bugs.json), each carrying the fix that resolved it.

Embeds with the same model Member 1's repository agent uses --
sentence-transformers' all-MiniLM-L6-v2 -- and searches a FAISS index built
over the historical issue text. When those packages (or their model weights)
aren't available, falls back to plain lexical similarity so the demo still
runs end to end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # same model as the repository agent
DEFAULT_TOP_K = 3
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "bugs.json"


@dataclass
class SimilarBug:
    """One historical bug retrieved as precedent for the current issue."""

    issue: str
    fix: str
    similarity: float
    source: str = "corpus"


def load_bugs(data_path: Path | str = DEFAULT_DATA_PATH) -> list[dict]:
    """Load the historical bug-fix corpus from a JSON file."""
    path = Path(data_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class BugRecommender:
    """Loads the historical bug corpus once and serves similarity search."""

    def __init__(self, data_path: Path | str = DEFAULT_DATA_PATH):
        self.bugs = load_bugs(data_path)
        self._model = None
        self._index = None
        self._backend = "lexical"
        self._build_index()

    def _build_index(self) -> None:
        if not self.bugs:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            embeddings = self._model.encode(
                [bug["issue"] for bug in self.bugs],
                convert_to_numpy=True,
            ).astype("float32")
            faiss.normalize_L2(embeddings)  # unit vectors -> inner product == cosine similarity

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._index = index
            self._backend = "faiss"
        except Exception:
            # sentence-transformers/faiss not installed, no model weights cached, etc.
            self._model = None
            self._index = None
            self._backend = "lexical"

    def _search_faiss(self, issue_text: str, top_k: int) -> list[SimilarBug]:
        import faiss

        query = self._model.encode([issue_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, len(self.bugs)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            bug = self.bugs[idx]
            results.append(SimilarBug(issue=bug["issue"], fix=bug["fix"], similarity=round(float(score), 2)))
        return results

    def _search_lexical(self, issue_text: str, top_k: int) -> list[SimilarBug]:
        from difflib import SequenceMatcher

        scored = [
            (SequenceMatcher(None, issue_text.lower(), bug["issue"].lower()).ratio(), bug)
            for bug in self.bugs
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SimilarBug(issue=bug["issue"], fix=bug["fix"], similarity=round(score, 2))
            for score, bug in scored[:top_k]
        ]

    def find_similar(self, issue_text: str, top_k: int = DEFAULT_TOP_K) -> list[SimilarBug]:
        """Retrieve the top-k historical bugs most similar to `issue_text`."""
        if not self.bugs:
            return []
        if self._backend == "faiss":
            return self._search_faiss(issue_text, top_k)
        return self._search_lexical(issue_text, top_k)


_default_recommender: BugRecommender | None = None


def get_default_recommender() -> BugRecommender:
    """Lazily build (and cache) the recommender over the default bug corpus."""
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = BugRecommender()
    return _default_recommender


def recommend_similar_bugs(issue_text: str, top_k: int = DEFAULT_TOP_K) -> list[SimilarBug]:
    """Convenience entry point using the cached default recommender."""
    return get_default_recommender().find_similar(issue_text, top_k)
