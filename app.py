import streamlit as st
import faiss
import numpy as np
import torch
import json
from transformers import CLIPProcessor, CLIPModel

st.title("🎥 Intelligent Video Search Engine")

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load index
index = faiss.read_index("video.index")
timestamps = np.load("timestamps.npy")

query = st.text_input("Enter search query")

if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # ✅ VERSION-SAFE TEXT EMBEDDING
    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        vec = text_outputs.pooler_output

    # Normalize
    vec = vec / vec.norm(dim=-1, keepdim=True)
    vec = vec.cpu().numpy().astype("float32")

    # Search
    D, I = index.search(vec, 5)

    results = []

    for rank, idx in enumerate(I[0]):

        sec = int(timestamps[idx])

        hrs = sec // 3600
        mins = (sec % 3600) // 60
        secs = sec % 60

        st.subheader(f"Result {rank+1}")
        st.write(f"Timestamp: {hrs:02}:{mins:02}:{secs:02}")
        st.image(f"frames/frame_{idx}.jpg")

        results.append({
            "rank": rank + 1,
            "timestamp": f"{hrs:02}:{mins:02}:{secs:02}",
            "frame": f"frames/frame_{idx}.jpg",
            "query": query
        })

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)