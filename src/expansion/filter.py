from typing import List, Optional

from sentence_transformers import SentenceTransformer, util


class RelevanceFilter:
    """
    Filters candidate papers by semantic similarity to a query string.

    Uses SentenceTransformers cosine similarity. Candidates below the
    similarity threshold are excluded; passing candidates are returned
    sorted descending by score.

    Usage:
        rf = RelevanceFilter()
        relevant = rf.filter(
            query="knowledge graph embedding",
            candidates=[{"paperId": "abc", "title": "...", "abstract": "..."}, ...],
            threshold=0.3,
        )
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Args:
            model_name: SentenceTransformers model identifier.
                        Default "all-MiniLM-L6-v2" — fast, good quality for
                        academic text similarity.
        """
        self._model = SentenceTransformer(model_name)

    def filter(
        self,
        query: str,
        candidates: List[dict],
        threshold: float = 0.3,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """
        Return candidates whose cosine similarity to query >= threshold.

        Args:
            query:      Search query string (same query used for seed()).
            candidates: List of paper dicts from get_papers_batch() API response.
                        Expected keys: paperId, title, abstract (all optional).
            threshold:  Minimum cosine similarity to include (0.0–1.0).
            top_k:      If set, return at most top_k candidates after filtering.

        Returns:
            Filtered list of candidate dicts, sorted descending by similarity.
            Original dicts are returned unchanged (not copied or modified).
            Returns [] if candidates is empty or none pass threshold.
        """
        if not candidates:
            return []

        texts = [
            (
                (d.get("title") or "") + " " + (d.get("abstract") or "")
            ).strip()
            for d in candidates
        ]

        embeddings = self._model.encode(
            [query] + texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        query_vec = embeddings[0]
        cand_vecs = embeddings[1:]

        scores = util.cos_sim(query_vec, cand_vecs)[0]  # shape (N,)

        scored = [
            (candidates[i], float(scores[i]))
            for i in range(len(candidates))
            if float(scores[i]) >= threshold
        ]

        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return [d for d, _ in scored]
