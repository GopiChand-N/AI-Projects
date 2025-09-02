import numpy as np

def cosine_scores(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    return doc_matrix @ query_vec

def find_top(question: str, embed_text, doc_embeddings: np.ndarray, image_paths: list[str], top_k: int = 5):
    if doc_embeddings is None or len(image_paths) == 0:
        return []
    if doc_embeddings.shape[0] != len(image_paths):
        raise ValueError(f"embeddings count ({doc_embeddings.shape[0]}) != image_paths count ({len(image_paths)})")
    q = embed_text(question)
    scores = cosine_scores(q, doc_embeddings)
    idx = np.argsort(-scores)[:top_k]
    return [(image_paths[i], float(scores[i])) for i in idx]