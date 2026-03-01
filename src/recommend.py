import pickle
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from difflib import get_close_matches

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def recommend_for_user(model: dict, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
    # ... your existing implementation ...
    pass  # Keep your original implementation here

def find_similar_books_by_book(model: dict, target_book_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
    """Find similar books with robust error handling"""
    try:
        item_map = model.get("item_map", {})
        item_ids = model.get("item_ids", [])
        
        if not item_map or not item_ids:
            return []
            
        if target_book_id not in item_map:
            return []
            
        idx = item_map[target_book_id]
        item_components = model.get("item_components")
        if item_components is None or item_components.size == 0:
            return []
            
        item_vec = item_components.T[idx]
        all_vecs = item_components.T
        
        # Safe normalization
        def norm(a):
            n = np.linalg.norm(a, axis=1)
            n = np.where(n == 0, 1.0, n)  # Avoid division by zero
            return a / n[:, None]
        
        all_norm = norm(all_vecs)
        item_norm = item_vec / (np.linalg.norm(item_vec) or 1.0)
        sims = all_norm.dot(item_norm)
        
        # Create pairs excluding the target book
        pairs = [
            (item_ids[i], float(sims[i])) 
            for i in range(len(item_ids)) 
            if i != idx and not np.isnan(sims[i])
        ]
        
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]
        
    except Exception as e:
        # Log error in production, for now just return empty list
        return []

def lookup_books(books_df: pd.DataFrame, q: str, max_results: int = 10) -> pd.DataFrame:
    q_lower = q.lower()
    sub = books_df[books_df["title"].str.lower().str.contains(q_lower, na=False)]
    if sub.empty:
        titles = books_df["title"].dropna().unique().tolist()
        matches = get_close_matches(q, titles, n=max_results, cutoff=0.5)
        sub = books_df[books_df["title"].isin(matches)]
    return sub.head(max_results)

def get_popular_books(model: Optional[dict], books: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Get popular books based on:
    1. Actual rating counts from model interactions (if available)
    2. Average rating fallback (if no model)
    """
    # Fallback when no model or interactions data
    if model is None or "interactions" not in model:
        if "average_rating" in books.columns:
            sorted_books = books.sort_values("average_rating", ascending=False)
        else:
            # Ultimate fallback - return first N books
            sorted_books = books
        return sorted_books.head(top_n).copy()
    
    # Count ratings per book from interactions
    book_counts: Dict[int, int] = {}
    for user_books in model["interactions"].values():
        for bid in user_books:
            book_counts[bid] = book_counts.get(bid, 0) + 1
    
    # Get top books by count
    top_books = sorted(book_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ids = [bid for bid, _ in top_books]
    
    # Filter and sort books
    popular_df = books[books["book_id"].isin(top_ids)].copy()
    popular_df["rating_count"] = popular_df["book_id"].map(book_counts)
    popular_df = popular_df.sort_values("rating_count", ascending=False)
    return popular_df.head(top_n)