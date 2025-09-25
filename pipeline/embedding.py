from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def clean_and_limit(self, skills_str, top_n):
        if not skills_str: return ""
        skills = [s.strip() for s in str(skills_str).split(",") if s.strip()]
        return ", ".join(skills[:top_n])

    def encode_batch(self, text_list, batch_size=64):
        embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding"):
            batch = text_list[i:i+batch_size]
            emb = self.model.encode(batch, normalize_embeddings=True)
            embeddings.append(emb)
        return np.vstack(embeddings)
