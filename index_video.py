import cv2
import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Video path
video_path = "sample_video/video.mp4"

os.makedirs("frames", exist_ok=True)

print("Loading model...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("❌ Cannot open video file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps * 5  # every 5 seconds

embeddings = []
timestamps = []

count = 0
idx = 0

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        frame_path = f"frames/frame_{idx}.jpg"
        cv2.imwrite(frame_path, frame)

        image = Image.open(frame_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        # ✅ FIXED INDENTATION + CLIP projection
        with torch.no_grad():
            image_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = image_outputs.pooler_output

            # Project to CLIP embedding space (512 dim)
            vec = model.visual_projection(pooled)

        # Normalize
        vec = vec / vec.norm(dim=-1, keepdim=True)

        embeddings.append(vec[0].cpu().numpy())
        timestamps.append(count / fps)

        print(f"Indexed frame {idx}")

        idx += 1

    count += 1

cap.release()

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

if len(embeddings) == 0:
    raise ValueError("❌ No embeddings created")

print("Embeddings shape:", embeddings.shape)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and timestamps
faiss.write_index(index, "video.index")
np.save("timestamps.npy", timestamps)

print("✅ Index Created Successfully")


