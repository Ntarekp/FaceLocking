"""
src/evaluate.py
Threshold tuning and evaluation using enrollment crops.

Assumptions:
- Enrollment crops exist under: data/enroll/<name>/*.jpg
- Crops are aligned (112x112)
- Uses ArcFaceEmbedderONNX from src.embed

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.embed import ArcFaceEmbedderONNX


@dataclass
class EvalConfig:
    """Configuration for the evaluation suite."""
    enroll_dir: Path = Path("data/enroll")
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80      # Cap for processing speed
    target_far: float = 0.01           # 1% False Acceptance Rate target
    thresholds: Tuple[float, float, float] = (0.10, 1.20, 0.01)  # start, end, step
    require_size: Tuple[int, int] = (112, 112)  # Expected alignment size


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity. Note: Embeddings are expected to be L2-normalized."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    # Since embeddings are L2-normalized, the dot product equals cosine similarity
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine distance as 1.0 - cosine similarity."""
    return 1.0 - cosine_similarity(a, b)


def list_people(cfg: EvalConfig) -> List[Path]:
    """Retrieves list of directories for enrolled individuals."""
    if not cfg.enroll_dir.exists():
        raise FileNotFoundError(
            f"Enroll dir not found: {cfg.enroll_dir}. Run enroll.py first."
        )
    return sorted([p for p in cfg.enroll_dir.iterdir() if p.is_dir()])


def _is_aligned_crop(img: np.ndarray, req: Tuple[int, int]) -> bool:
    """Verifies image dimensions match requirement."""
    h, w = img.shape[:2]
    return (w, h) == (int(req[0]), int(req[1]))


def load_embeddings_for_person(
    embedder: ArcFaceEmbedderONNX,
    person_dir: Path,
    cfg: EvalConfig,
) -> List[np.ndarray]:
    """Loads and generates embeddings for a specific person's images."""
    imgs = sorted(list(person_dir.glob("*.jpg")))[:cfg.max_imgs_per_person]
    embs: List[np.ndarray] = []

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if cfg.require_size is not None and not _is_aligned_crop(img, cfg.require_size):
            continue

        res = embedder.embed(img)
        embs.append(res.embedding)

    return embs


def pairwise_distances(
    embs_a: List[np.ndarray], 
    embs_b: List[np.ndarray], 
    same: bool
) -> List[float]:
    """Computes distances between pairs of embeddings."""
    dists: List[float] = []
    if same:
        # Intra-class (Same person)
        for i in range(len(embs_a)):
            for j in range(i + 1, len(embs_a)):
                dists.append(cosine_distance(embs_a[i], embs_a[j]))
    else:
        # Inter-class (Different people)
        for ea in embs_a:
            for eb in embs_b:
                dists.append(cosine_distance(ea, eb))
    return dists


def sweep_thresholds(genuine: np.ndarray, impostor: np.ndarray, cfg: EvalConfig):
    """Calculates FAR and FRR over a range of distance thresholds."""
    t0, t1, step = cfg.thresholds
    thresholds = np.arange(t0, t1 + 1e-9, step, dtype=np.float32)

    results = []
    for thr in thresholds:
        # FAR: impostor accepted (distance <= threshold)
        far = float(np.mean(impostor <= thr)) if impostor.size else 0.0
        # FRR: genuine rejected (distance > threshold)
        frr = float(np.mean(genuine > thr)) if genuine.size else 0.0
        results.append((float(thr), far, frr))
    return results


def describe(arr: np.ndarray) -> str:
    """Returns a statistical summary of the distance array."""
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size:4d} | mean={arr.mean():.3f} | std={arr.std():.3f} | "
        f"p05={np.percentile(arr, 5):.3f} | p50={np.percentile(arr, 50):.3f} | "
        f"p95={np.percentile(arr, 95):.3f}"
    )


def main():
    cfg = EvalConfig()

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    people_dirs = list_people(cfg)
    if len(people_dirs) < 1:
        print("No enrolled people found.")
        return

    # 1. Load and process embeddings
    per_person: Dict[str, List[np.ndarray]] = {}
    for pdir in people_dirs:
        name = pdir.name
        embs = load_embeddings_for_person(embedder, pdir, cfg)
        if len(embs) >= cfg.min_imgs_per_person:
            per_person[name] = embs
        else:
            print(f"Skipping {name}: insufficient aligned crops (found {len(embs)}).")

    names = sorted(per_person.keys())
    if len(names) < 1:
        print("Not enough data to evaluate. Enroll more samples.")
        return

    # 2. Calculate Distances
    genuine_all: List[float] = []
    for name in names:
        genuine_all.extend(pairwise_distances(per_person[name], per_person[name], same=True))

    impostor_all: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor_all.extend(pairwise_distances(per_person[names[i]], per_person[names[j]], same=False))

    genuine = np.array(genuine_all, dtype=np.float32)
    impostor = np.array(impostor_all, dtype=np.float32)

    # 3. Print Results
    print("\n=== Distance Distributions (Cosine Distance) ===")
    print(f"Genuine (Same):   {describe(genuine)}")
    print(f"Impostor (Diff):  {describe(impostor)}")

    results = sweep_thresholds(genuine, impostor, cfg)

    # 4. Find optimal threshold
    best = None
    for thr, far, frr in results:
        if far <= cfg.target_far:
            if best is None or frr < best[2]:
                best = (thr, far, frr)

    print("\n=== Threshold Sweep ===")
    stride = max(1, len(results) // 10)
    for thr, far, frr in results[::stride]:
        print(f"thr={thr:.2f}  FAR={far*100:5.2f}%  FRR={frr*100:5.2f}%")

    if best is not None:
        thr, far, frr = best
        print(f"\n[Best Recommendation for target FAR {cfg.target_far*100:.1f}%]")
        print(f"Distance Threshold:   {thr:.2f}")
        print(f"FAR Result:           {far*100:.2f}%")
        print(f"FRR Result:           {frr*100:.2f}%")
        print(f"Cosine Sim equiv:     ~{1.0 - thr:.3f}")
    else:
        print(f"\nNo threshold met FAR <= {cfg.target_far*100:.1f}%.")

    print()


if __name__ == "__main__":
    main()
