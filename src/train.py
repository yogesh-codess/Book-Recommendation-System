import argparse
from collections import Counter
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def gather_top_entities(ratings_path, max_users=20000, max_items=5000, chunksize=200000):
    user_counts = Counter()
    item_counts = Counter()
    for chunk in pd.read_csv(ratings_path, usecols=[0,1], header=0, names=None, chunksize=chunksize):
        # assume columns are user_id, book_id, rating (or similar) - use the first two numeric columns
        if chunk.shape[1] >= 2:
            u = chunk.iloc[:, 0].astype(int)
            i = chunk.iloc[:, 1].astype(int)
            user_counts.update(u.values)
            item_counts.update(i.values)
    top_users = [u for u, _ in user_counts.most_common(max_users)]
    top_items = [b for b, _ in item_counts.most_common(max_items)]
    return set(top_users), set(top_items)


def build_matrix(ratings_path, users_keep, items_keep, chunksize=200000):
    rows = []
    for chunk in pd.read_csv(ratings_path, header=0, chunksize=chunksize):
        # try to find numeric user and item columns
        cols = chunk.select_dtypes(include=["number"]).columns
        if len(cols) < 2:
            continue
        user_col, item_col = cols[0], cols[1]
        sub = chunk[chunk[user_col].isin(users_keep) & chunk[item_col].isin(items_keep)][[user_col, item_col, *(cols[2:3])]]
        # rating if exists
        if sub.shape[1] >= 3:
            ratings = sub.iloc[:, 2]
        else:
            ratings = pd.Series(1, index=sub.index)
        for u, b, r in zip(sub.iloc[:, 0].astype(int), sub.iloc[:, 1].astype(int), ratings):
            rows.append((u, b, float(r)))
    df = pd.DataFrame(rows, columns=["user_id", "book_id", "rating"]) if rows else pd.DataFrame(columns=["user_id", "book_id", "rating"])
    return df


def train(ratings_path, books_path, out_path, n_components=50, max_users=20000, max_items=5000):
    print("Scanning ratings to select top users/items...")
    users_keep, items_keep = gather_top_entities(ratings_path, max_users=max_users, max_items=max_items)
    print(f"Selected {len(users_keep)} users and {len(items_keep)} items")
    print("Building filtered interaction matrix...")
    df = build_matrix(ratings_path, users_keep, items_keep)
    if df.empty:
        raise RuntimeError("No rating rows after filtering - check file format and parameters")

    # create index mappings
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["book_id"].unique())
    user_map = {u: idx for idx, u in enumerate(unique_users)}
    item_map = {b: idx for idx, b in enumerate(unique_items)}

    rows = [user_map[u] for u in df["user_id"]]
    cols = [item_map[b] for b in df["book_id"]]
    data = df["rating"].values.astype(np.float32)

    mat = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))

    print("Fitting TruncatedSVD...")
    svd = TruncatedSVD(n_components=min(n_components, mat.shape[1]-1 or 1), random_state=42)
    user_factors = svd.fit_transform(mat)
    item_components = svd.components_  # shape (k, n_items)

    # Save model and metadata
    print(f"Saving model to {out_path}")
    books = pd.read_csv(books_path)
    model = {
        "svd": svd,
        "user_map": user_map,
        "item_map": item_map,
        "user_ids": unique_users,
        "item_ids": unique_items,
        "books": books,
        "user_factors": user_factors,
        "item_components": item_components,
        # store interactions to filter already-seen items
        "interactions": df.groupby("user_id")["book_id"].apply(set).to_dict(),
    }
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple SVD recommender")
    parser.add_argument("--ratings", default="data/ratings.csv")
    parser.add_argument("--books", default="data/books.csv")
    parser.add_argument("--out", default="model/svd_model.pkl")
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--max-users", type=int, default=20000)
    parser.add_argument("--max-items", type=int, default=5000)
    args = parser.parse_args()
    train(args.ratings, args.books, args.out, n_components=args.n_components, max_users=args.max_users, max_items=args.max_items)