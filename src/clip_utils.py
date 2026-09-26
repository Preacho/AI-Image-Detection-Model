import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
import tensorflow as tf

CLIP_MODEL = "ViT-B-32"
CLIP_CKPT  = "laion2b_s34b_b79k"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_clip():
    """Load CLIP once and return (model, preprocess)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_CKPT
    )
    return model.to(CLIP_DEVICE).eval(), preprocess

def _collect_paths(directory, image_size=(300, 300), color_mode="grayscale"):
    """Return file paths in the SAME order image_dataset_from_directory yields."""
    data = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=64,
        image_size=image_size,
        color_mode=color_mode,
        shuffle=False,
    )
    return list(data.file_paths)

@torch.no_grad()
def _embed_paths(paths, model, preprocess, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP"):
        batch = []
        for p in paths[i:i + batch_size]:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(preprocess(img))
            except Exception as e:
                print(f"[WARN] {p}: {e}")
                batch.append(torch.zeros(3, 224, 224))
        x = torch.stack(batch).to(CLIP_DEVICE)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)   # L2-normalize
        embs.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)

def clip_features(directory, cache_path,
                  image_size=(300, 300), color_mode="grayscale"):
    """
    Extract CLIP embeddings for every image under <directory>/<class>/*.
    Rows are in the same order as load_data() returns them.
    Cached to `cache_path` so re-runs are instant.
    """
    if os.path.exists(cache_path):
        print(f"[CLIP] Loading cache: {cache_path}")
        return np.load(cache_path)["emb"]

    print(f"Executing [CLIP] FROM {directory}...")
    paths = _collect_paths(directory, image_size=image_size,
                           color_mode=color_mode)

    model, preprocess = _load_clip()

    embs = _embed_paths(paths, model, preprocess)

    np.savez(cache_path, emb=embs)
    return embs