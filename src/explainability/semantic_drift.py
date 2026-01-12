import numpy as np

from src.embeddings.similarity import l2_normalize


def cosine(a, b):
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


def compute_drift(plan_emb, output_emb):
    return 1.0 - cosine(plan_emb, output_emb)
