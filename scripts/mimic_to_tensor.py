import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

OUTPUT_BASE = "/well/papiez/users/hri611/python/data-centric-bias/data/MIMIC"
PREFIX = "/gpfs3/well/papiez/shared/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/" # old path you will replace

os.makedirs(OUTPUT_BASE, exist_ok=True)

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor(),  # Returns [1, H, W] for grayscale
        transforms.ConvertImageDtype(torch.float32)
        ])

# for df_path in [os.path.join(OUTPUT_BASE,'train_labels.csv.gz'), os.path.join(OUTPUT_BASE,'val_labels.csv.gz'), os.path.join(OUTPUT_BASE,'test_labels.csv.gz')]:
for df_path in [os.path.join(OUTPUT_BASE,'val_labels.csv.gz'), os.path.join(OUTPUT_BASE,'test_labels.csv.gz')]:


    df = pd.read_csv(df_path)

    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["path"]
        rel_path = os.path.relpath(img_path, PREFIX)  # gives pxx/.../filename.jpg
        rel_path_pt = os.path.splitext(rel_path)[0] + ".pt"

        output_path = os.path.join(OUTPUT_BASE, 'images', rel_path_pt)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = Image.open(img_path).convert("L")
        tensor = transform(img)  # [1, 256, 256]

        torch.save(tensor, output_path)
        new_paths.append(output_path)

    df["tensor_path"] = new_paths
    df.to_csv(df_path, index=False)
