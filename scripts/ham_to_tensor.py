import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Base output folder
OUTPUT_BASE = "/well/papiez/users/hri611/python/data-centric-bias/data/HAM10000/images"

# Transformation: resize to 256x256 and convert to float tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.PILToTensor(),            # Converts to [C, H, W]
    transforms.ConvertImageDtype(torch.float32)
])

# Read metadata CSV
df_path = "/well/papiez/users/hri611/python/data-centric-bias/data/HAM10000/processed_metadata.csv"
df = pd.read_csv(df_path)

new_paths = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["Path"]  # original JPG path
    
    # # Make output path for tensor
    # filename = os.path.splitext(os.path.basename(img_path))[0] + ".pt"
    # output_path = os.path.join(OUTPUT_BASE, filename)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # img = Image.open(img_path).convert("RGB")  # 3 channels
    # tensor = transform(img)                    # [3, 256, 256]

    # torch.save(tensor, output_path)

    os.remove(img_path)

    # new_paths.append(output_path)

# Add new column to dataframe and save
# df["tensor_path"] = new_paths
# df.to_csv(df_path, index=False)

print("Done! All images converted to 256x256 tensors.")
