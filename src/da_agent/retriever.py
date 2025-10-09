# src/da_agent/retriever.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

class SimpleRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path='data/faiss.index', docs_path='data/docs.pkl'):
        self.embed_model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.docs_path = docs_path
        self.index = None
        self.docs = []
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path,'rb') as f:
                self.docs = pickle.load(f)

    def build(self, docs):
        """
        docs: list of strings
        """
        self.docs = docs
        embs = self.embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embs)
        index.add(embs)
        faiss.write_index(index, self.index_path)
        with open(self.docs_path,'wb') as f:
            pickle.dump(docs, f)
        self.index = index

    def retrieve(self, query, k=5):
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        hits = [self.docs[int(idx)] for idx in I[0] if idx >= 0]
        return hits
