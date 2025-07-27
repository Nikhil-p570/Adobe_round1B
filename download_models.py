from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading Bi-Encoder model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading Cross-Encoder model...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("All models have been downloaded to your local cache.")