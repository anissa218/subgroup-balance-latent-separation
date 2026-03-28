import os
import glob
import torch
import pandas as pd
from transformers import SiglipImageProcessor, SiglipModel
from PIL import Image
import time

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

processor = SiglipImageProcessor.from_pretrained(
    "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
)
model = SiglipModel.from_pretrained(
    "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli",
    torch_dtype=dtype
).to(device)
print("loaded model")

# -------------------------
# Walk over CSVs
# -------------------------
csv_files = glob.glob(
    "/well/papiez/users/hri611/python/data-centric-bias/data/MIMICFM/**/*.csv",
    recursive=True,
)
PREFIX = "/gpfs3/well/papiez/shared/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
OUTPUT_BASE = "/well/papiez/users/hri611/python/data-centric-bias/data/MIMIC"

BATCH_SIZE = 32

for csv_path in csv_files:
    print(f"Processing {csv_path} ...")
    df = pd.read_csv(csv_path)

    embed_paths = []
    batch_images = []
    batch_embed_paths = []

    for path in df["path"]:
        rel_path = os.path.relpath(path, PREFIX)
        rel_path_pt = os.path.splitext(rel_path)[0] + "_embed.pt"
        embed_path = os.path.join(OUTPUT_BASE, "images", rel_path_pt)
        embed_paths.append(embed_path)

        if os.path.exists(embed_path):
            continue

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        batch_images.append(image)
        batch_embed_paths.append(embed_path)

        # Process batch when full
        if len(batch_images) == BATCH_SIZE:
            start = time.time()
            inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

            with torch.no_grad():
                outputs = model.vision_model(**inputs)

            embeddings = outputs.pooler_output.cpu()  # [B, hidden_dim]

            batch_time = time.time() - start
            print(f"Processed batch of {len(batch_images)} in {batch_time:.2f}s "
                  f"({batch_time/len(batch_images):.3f}s/image)")

            # Save individually
            for emb, out_path in zip(embeddings, batch_embed_paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(emb, out_path)

            batch_images, batch_embed_paths = [], []

    # Process leftover images
    if batch_images:
        inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.no_grad():
            outputs = model.vision_model(**inputs)
        embeddings = outputs.pooler_output.cpu()
        for emb, out_path in zip(embeddings, batch_embed_paths):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(emb, out_path)

    df["embed_path"] = embed_paths
    df.to_csv(csv_path, index=False)
