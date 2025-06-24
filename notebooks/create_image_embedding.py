import os
import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
from tqdm import tqdm

# Modell és processor betöltése
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)
processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

# Képek mappa
base_path = "dataset/train" # Itt van: https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset

# Képek beolvasása
image_paths = []
image_labels = []
for folder in sorted(os.listdir(base_path), key=lambda x: int(x)):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder_path, img_file))
                image_labels.append(folder)

# Embedek számolása
image_features = []
image_infos = []

for img_path, label in tqdm(zip(image_paths, image_labels), total=len(image_paths)):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        image_features.append(feat.cpu())
        image_infos.append({"path": img_path, "label": label})
    except Exception as e:
        print(f"Hiba: {img_path} – {e}")

# Egyesítés és mentés
image_features = torch.cat(image_features, dim=0)
torch.save(image_features, "../data/image_embeddings_siglip.pt")
torch.save(image_infos, "../data/image_info_siglip.pt")