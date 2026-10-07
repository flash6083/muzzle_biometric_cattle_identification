from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_eer(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float64)
    if np.unique(labels_array).size != 2:
        raise ValueError("EER requires both genuine and impostor pairs")

    fpr, tpr, thresholds = roc_curve(labels_array, scores_array)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    return eer, float(thresholds[index])


def compute_tar_at_far(
    labels: Sequence[int],
    scores: Sequence[float],
    far_values: Sequence[float] = (0.01, 0.1),
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    results = {}
    for far in far_values:
        eligible = np.flatnonzero(fpr <= far)
        results[str(far)] = float(tpr[eligible[-1]]) if eligible.size else 0.0
    return results


def evaluate_verification(
    pairs: Sequence[dict[str, object]],
    embeddings: dict[str, np.ndarray],
    far_values: Sequence[float] = (0.01, 0.1),
) -> dict[str, object]:
    labels = []
    scores = []
    for pair in pairs:
        path_a = str(pair["image_a"])
        path_b = str(pair["image_b"])
        if path_a not in embeddings or path_b not in embeddings:
            raise KeyError(f"Missing image embedding for pair: {path_a}, {path_b}")
        embedding_a = embeddings[path_a]
        embedding_b = embeddings[path_b]
        score = float(np.dot(embedding_a, embedding_b))
        scores.append(score)
        labels.append(int(pair["label"]))

    positive_count = int(sum(labels))
    negative_count = len(labels) - positive_count
    result: dict[str, object] = {
        "num_pairs": len(labels),
        "genuine_pairs": positive_count,
        "impostor_pairs": negative_count,
        "eer": None,
        "eer_threshold": None,
        "roc_auc": None,
        "tar_at_far": {},
    }
    if positive_count == 0 or negative_count == 0:
        return result

    eer, eer_threshold = compute_eer(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    result.update(
        {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "roc_auc": float(auc(fpr, tpr)),
            "tar_at_far": compute_tar_at_far(labels, scores, far_values),
        }
    )
    return result