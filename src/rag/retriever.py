# src/rag/retriever.py
import os
import glob
import torch
from sentence_transformers import SentenceTransformer, util

class RAGRetriever:
    def __init__(self, knowledge_dir="data/knowledge_base", model_name="all-MiniLM-L6-v2"):
        self.knowledge_dir = knowledge_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model ({model_name}) on {self.device}...")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        self.chunks = []
        self.embeddings = None
        self.load_and_index_documents()

    def load_and_index_documents(self):
        """Loads all .txt files from data/knowledge_base and builds document embeddings."""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)
            # Create a sample document if empty
            sample_file = os.path.join(self.knowledge_dir, "sample.txt")
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("Mini-LLM is a GPT-style autoregressive language model built using PyTorch.\n")
                f.write("The capital of France is Paris. It is known for its culture, art, and the Eiffel Tower.\n")
                f.write("Photosynthesis is the process used by plants to convert light energy into chemical energy.\n")
            print(f"Created default knowledge base document at {sample_file}")

        txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        all_lines = []

        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                all_lines.extend(lines)

        if not all_lines:
            print("Warning: Knowledge base is empty.")
            return

        self.chunks = all_lines
        print(f"Indexing {len(self.chunks)} text chunks from knowledge base...")
        self.embeddings = self.encoder.encode(self.chunks, convert_to_tensor=True, show_progress_bar=False)
        print("Knowledge base indexing complete!")

    def retrieve(self, query, top_k=2):
        """Retrieves top_k most relevant text chunks for a given query."""
        if not self.chunks or self.embeddings is None:
            return ""

        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.chunks)))

        retrieved_text = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score.item() > 0.25:  # Relevance threshold filtering
                retrieved_text.append(self.chunks[idx.item()])

        return " ".join(retrieved_text)