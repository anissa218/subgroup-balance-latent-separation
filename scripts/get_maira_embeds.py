from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(task="image-feature-extraction", model="microsoft/rad-dino-maira-2", pool=False)
csv_files = glob.glob(
    "/well/papiez/users/hri611/python/data-centric-bias/data/MIMICFM2_Small/**/*.csv",
    recursive=True,
)
PREFIX = "/gpfs3/well/papiez/shared/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
OUTPUT_BASE = "/well/papiez/users/hri611/python/data-centric-bias/data/MIMIC"

BATCH_SIZE = 32

pipe = pipeline(task="image-feature-extraction", model="microsoft/rad-dino-maira-2", pool=False)

for csv_path in csv_files:
    print(f"Processing {csv_path} ...")
    df = pd.read_csv(csv_path)

    embed_paths = []
    batch_images = []
    batch_embed_paths = []

    for path in df["path"]:
        rel_path = os.path.relpath(path, PREFIX)
        rel_path_pt = os.path.splitext(rel_path)[0] + "_embed2.pt"
        embed_path = os.path.join(OUTPUT_BASE, "images", rel_path_pt)
        embed_paths.append(embed_path)

        if os.path.exists(embed_path):
            continue

        batch_images.append(path)
        batch_embed_paths.append(embed_path)

        # Process batch when full
        if len(batch_images) == BATCH_SIZE:
            start = time.time()
            patch_features = pipe(batch_images)

            batch_time = time.time() - start
            print(f"Processed batch of {len(batch_images)} in {batch_time:.2f}s "
                  f"({batch_time/len(batch_images):.3f}s/image)")

            # Save individually
            for i,out_path in enumerate(batch_embed_paths):
                emb_list = patch_features[i][0][0] # get cls token (first token)
                emb = torch.tensor(emb_list)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(emb, out_path)

            batch_images, batch_embed_paths = [], []

    # Process leftover images
    if batch_images:
        patch_features = pipe(batch_images)
        for i,out_path in enumerate(batch_embed_paths):
            emb_list = patch_features[i][0][0] # get cls token (first token)
            emb = torch.tensor(emb_list)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(emb, out_path)

    df["embed_path"] = embed_paths
    df.to_csv(csv_path, index=False)