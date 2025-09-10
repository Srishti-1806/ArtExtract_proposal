import os, csv, math
import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence, Optional
from dataclasses import dataclass
from PIL import Image, ImageOps
from collections import defaultdict

@dataclass
class SearchResult:
    ids: np.ndarray
    scores: np.ndarray


def load_ids(ids_csv: str) -> np.ndarray:
    return np.loadtxt(ids_csv, dtype=str, delimiter=",")


def load_filenames(
    filenames_csv: str,
    base_dirs: str | list[str] | tuple[str, ...] | None = None,
    prefer_order: tuple[str, ...] = ("rgb_images", "ms_masks"),
) -> np.ndarray:
    """Load filenames from a CSV file and resolve their paths.
    Args:
        filenames_csv (str): Path to the CSV file containing filenames.
        base_dirs (str | list[str] | tuple[str, ...] | None, optional): Base directories to search for files. Defaults to None.
        prefer_order (tuple[str, ...], optional): Preferred order of directories to search. Defaults to ("rgb_images", "ms_masks").

    Raises:
        FileNotFoundError: If a file cannot be found.

    Returns:
        np.ndarray: Resolved file paths.
    """
    raw_paths: list[str] = []
    with open(filenames_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if len(rows) > 0 and len(rows[0]) >= 2 and rows[0][1].strip():
            iterable = rows[1:] if rows[0][0].lower().startswith("idx") else rows
            for r in iterable:
                if len(r) >= 2 and r[1].strip():
                    raw_paths.append(r[1].strip())
        else:
            iterable = rows[1:] if rows and rows[0] and rows[0][0].lower() in ("path", "paths") else rows
            for r in iterable:
                if len(r) >= 1 and r[0].strip():
                    raw_paths.append(r[0].strip())

    if base_dirs is None:
        return np.array([os.path.normpath(p) for p in raw_paths], dtype=object)

    if isinstance(base_dirs, (str, os.PathLike)):
        base_dirs = [str(base_dirs)]
    else:
        base_dirs = [str(b) for b in base_dirs]

    name_to_paths: dict[str, list[str]] = defaultdict(list)
    for root in base_dirs:
        for dirpath, _, files in os.walk(root):
            for fname in files:
                name_to_paths[fname.lower()].append(os.path.normpath(os.path.join(dirpath, fname)))

    resolved: list[str] = []
    for p in raw_paths:
        if os.path.isabs(p) and os.path.exists(p):
            resolved.append(os.path.normpath(p))
            continue
        if os.path.exists(p):
            resolved.append(os.path.normpath(p))
            continue

        basename = os.path.basename(p).lower()
        candidates = name_to_paths.get(basename, [])

        if not candidates:
            raise FileNotFoundError(f"Cannot resolve path for '{p}'. "
                                    f"Search roots={base_dirs}")

        if len(candidates) == 1:
            resolved.append(candidates[0])
            continue

        chosen = None
        for key in prefer_order:
            for c in candidates:
                if key in c.replace("\\", "/"):
                    chosen = c
                    break
            if chosen:
                break
        resolved.append(chosen or candidates[0])

    return np.array(resolved, dtype=object)

def build_name_to_path(paths: Sequence[str] | np.ndarray) -> dict[str, str]:
    import os
    return {os.path.basename(str(p)): str(p) for p in paths}

def ensure_grid(n: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _safe_open(path, size=(256, 256), keep_gray=True, robust=True):
    img = Image.open(path)

    # If grayscale image and we want to keep it as grayscale
    if keep_gray and img.mode in ("I;16", "I", "F", "L"):
        img = img.resize(size, resample=Image.NEAREST)
        arr = np.array(img).astype(np.float32)

        if robust:
            p2, p98 = np.percentile(arr, (2, 98))
            if p98 > p2:
                arr = (arr - p2) / (p98 - p2)
            else:
                mn, mx = arr.min(), arr.max()
                arr = (arr - mn) / (mx - mn + 1e-6)
        else:
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn + 1e-6)
        return arr

    # Process RGB images
    img = img.convert("RGB").resize(size, resample=Image.BILINEAR)
    return np.array(img)  # shape: (H,W,3), uint8

def visualize_query_results(
    query_image_path: str,
    result: "SearchResult",
    id_to_path: list[str] | np.ndarray | dict[str, str],
    title: str = "Retrieval Results",
    max_width: int = 256,
):
    query_img = _safe_open(query_image_path, size=(max_width, max_width))

    K = len(result.ids)
    rows, cols = ensure_grid(K)

    plt.figure(figsize=(3 + 3*cols, 3*rows))

    # Query
    ax_q = plt.axes([0.02, 0.1, 0.25, 0.8])
    if query_img.ndim == 2:
        ax_q.imshow(query_img, cmap="viridis", interpolation="nearest")
    else:
        ax_q.imshow(query_img)
    ax_q.set_title(f"Query", fontsize=12)
    ax_q.axis("off")

    # Grid
    grid_left = 0.30
    grid_width = 0.68
    grid_height = 0.75
    cell_w = grid_width / cols
    cell_h = grid_height / rows

    for i, (rid, score) in enumerate(zip(result.ids, result.scores)):
        if isinstance(rid, (np.integer, int)) or str(rid).isdigit():
            if isinstance(id_to_path, dict):
                raise TypeError("id_to_path is a dict but rid is int; provide a sequence for index lookup.")
            path = id_to_path[int(rid)]
        else:
            name = os.path.basename(str(rid))
            if isinstance(id_to_path, dict) and name in id_to_path:
                path = id_to_path[name]
            else:
                path = str(rid)

        img = _safe_open(path, size=(max_width, max_width))

        r = i // cols
        c = i % cols
        left = grid_left + c * cell_w
        bottom = 0.1 + (rows - 1 - r) * cell_h
        ax = plt.axes([left, bottom, cell_w*0.9, cell_h*0.9])
        if img.ndim == 2:
            ax.imshow(img, cmap="viridis", interpolation="nearest")
        else:
            ax.imshow(img)
        ax.set_title(f"id={rid}", fontsize=10)
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.show()
    
def visualize_by_index(
    query_vec: np.ndarray,
    X: np.ndarray,
    ids: np.ndarray,
    id_to_path: Sequence[str] | np.ndarray | dict[str, str],
    index,            # faiss.Index
    topk: int = 8,
    query_image_path: Optional[str] = None,
    metric: str = "ip",   # "ip" or "l2"
    title: str = "Retrieval Results",
):
    """Visualize retrieval results given a query vector and a FAISS index.
    Args:
        query_vec (np.ndarray): Query feature vector.
        X (np.ndarray): Feature matrix used to build the index.
        ids (np.ndarray): Array of ids corresponding to the feature matrix.
        id_to_path (Sequence[str] | np.ndarray | dict[str, str]): Mapping from ids to image paths.
        index: FAISS index object.
        topk (int, optional): Number of top results to retrieve. Defaults to 8.
        query_image_path (Optional[str], optional): Path to the query image. If None, it will be inferred. Defaults to None.
        metric (str, optional): Similarity metric, either "ip" (inner product) or "l2" (Euclidean). Defaults to "ip".
        title (str, optional): Title of the plot. Defaults to "Retrieval Results".
    """
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32)

    D, I = index.search(query_vec, topk)
    hit_ids = ids[I[0]]
    hit_scores = D[0]

    if query_image_path is None:
        qid = ids[I[0][0]]
        if isinstance(qid, (np.integer, int)) or str(qid).isdigit():
            if isinstance(id_to_path, dict):
                raise TypeError("id_to_path is a dict but qid is int; provide a sequence for index lookup.")
            query_image_path = id_to_path[int(qid)]
        else:
            name = os.path.basename(str(qid))
            if isinstance(id_to_path, dict) and name in id_to_path:
                query_image_path = id_to_path[name]
            else:
                # Fallback: if qid is already a full path, use it; otherwise, fall back to the first available path
                query_image_path = (
                    str(qid) if os.path.exists(str(qid))
                    else (next(iter(id_to_path.values())) if isinstance(id_to_path, dict) else id_to_path[0])
                )

    visualize_query_results(
        query_image_path,
        SearchResult(ids=hit_ids, scores=hit_scores),
        id_to_path=id_to_path,
        title=title,
    )

