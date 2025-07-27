# save_models.py
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

print("Creating local_models directory...")
os.makedirs("local_models", exist_ok=True)

bi_encoder_path = "./local_models/bi-encoder"
cross_encoder_path = "./local_models/cross-encoder"

print("Downloading and saving Bi-Encoder model...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
bi_encoder.save(bi_encoder_path)

print("Downloading and saving Cross-Encoder model...")
# Note: CrossEncoder doesn't have a simple .save(), so we save its underlying transformer model.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_encoder.save(cross_encoder_path)


print(f"Models saved successfully to {os.path.abspath('local_models')}")