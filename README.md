# Book recommendation — training and Streamlit app

Quick steps

1. Create a virtual env and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the model (this reads `data/ratings.csv` and `data/books.csv`):

```bash
python -m src.train --ratings data/ratings.csv --books data/books.csv --out model/svd_model.pkl
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes
- The trainer limits memory by selecting the most active users and items (configurable with `--max-users` and `--max-items`).
- The model is a simple TruncatedSVD approximation; it's fast and easy to run locally.
