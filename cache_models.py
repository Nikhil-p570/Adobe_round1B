# cache_models.py
from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading and caching models...")

# Models used in the main script
bi_encoder_name = "all-MiniLM-L6-v2"
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# These lines download the models to the container's cache
SentenceTransformer(bi_encoder_name)
CrossEncoder(cross_encoder_name)

print("Models cached successfully.")