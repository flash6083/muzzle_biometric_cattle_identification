from __future__ import annotations

import random
from itertools import combinations, product
from pathlib import Path
from typing import Iterable


def generate_verification_pairs(
    image_paths: Iterable[Path],
    max_impostor_pairs: int = 10000,
    seed: int = 42,
) -> list[dict[str, object]]:
    images_by_identity: dict[str, list[Path]] = {}
    for path in sorted(Path(image_path) for image_path in image_paths):
        images_by_identity.setdefault(path.parent.name, []).append(path)

    pairs: list[dict[str, object]] = []
    for identity, paths in sorted(images_by_identity.items()):
        for image_a, image_b in combinations(paths, 2):
            pairs.append(
                {
                    "image_a": str(image_a),
                    "image_b": str(image_b),
                    "label": 1,
                    "type": "genuine",
                }
            )

    impostor_pairs = []
    identities = sorted(images_by_identity)
    for identity_a, identity_b in combinations(identities, 2):
        for image_a, image_b in product(
            images_by_identity[identity_a], images_by_identity[identity_b]
        ):
            impostor_pairs.append(
                {
                    "image_a": str(image_a),
                    "image_b": str(image_b),
                    "label": 0,
                    "type": "impostor",
                }
            )

    if max_impostor_pairs > 0 and len(impostor_pairs) > max_impostor_pairs:
        impostor_pairs = random.Random(seed).sample(impostor_pairs, max_impostor_pairs)
    pairs.extend(impostor_pairs)
    random.Random(seed).shuffle(pairs)
    return pairs
