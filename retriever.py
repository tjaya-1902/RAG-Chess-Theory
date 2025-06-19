import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# File paths for saved index and embeddings
CSV_FILE = "Chess_Chunks.csv"
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss_index.index"

# Load chunks
df = pd.read_csv(CSV_FILE, encoding="utf-8")
df.dropna(subset=["content"], inplace=True)
texts = df["content"].tolist()

# Load embedding model
print("Loading Sentence Transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence Transformer loaded!")

# Load or create embeddings + index 
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
    print("Loading saved FAISS index and embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)
    index = faiss.read_index(INDEX_FILE)
else:
    print("Creating new embeddings and FAISS index...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype('float32')
    np.save(EMBEDDINGS_FILE, embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

# Retrieval function ===
def retrieve_chunks(query, top_k=3):
    query_vec = model.encode([query]).astype('float32')
    _, indices = index.search(query_vec, top_k)
    return df.iloc[indices[0]]


def strip_move_numbers(pgn):
    """Remove turn numbers like '1.', '2.' from PGN."""
    return re.sub(r"\d+\.", "", pgn).strip()

def find_opening_by_moves(user_pgn, max_depth=8):
    """
    Match user's opening to the openings chunks.
    Return the best match from the 'moves' column where topic == 'openings'.
    """

    # Remove turn numbers and get first N moves (N = max_depth)
    clean_user_moves = strip_move_numbers(user_pgn).split()
    truncated_user_moves = " ".join(clean_user_moves[:max_depth])

    # Filter only openings
    openings_df = df[df["category"] == "Opening"]

    for _, row in openings_df.iterrows():
        opening_pgn = row.get("moves", "")
        clean_opening_moves = strip_move_numbers(opening_pgn).split()
        truncated_opening_moves = " ".join(clean_opening_moves[:max_depth])

        if truncated_user_moves == truncated_opening_moves:
            return row.get("topic"), row.get("content", "Opening found, but no description provided.")

    return "Opening not recognized in the databank."