import streamlit as st
import pandas as pd
import os
import numpy as np

from src.recommend import (
    load_model,
    get_popular_books,
    find_similar_books_by_book,
    lookup_books
)

MODEL_PATH = os.environ.get("SVD_MODEL_PATH", "model/svd_model.pkl")

@st.cache_resource
def load():
    """Load recommendation model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning(f"Model file not found at {MODEL_PATH}")
            return None
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def show_book(df_row):
    """Display book information with details below image"""
    if df_row is None or df_row.empty:
        st.warning("Book information unavailable")
        return
    
    try:
        # Extract values safely with defaults
        title = str(df_row.get("title", "Unknown Title")).strip()
        authors = str(df_row.get("authors", "Unknown Author")).strip()
        image_url = str(df_row.get("image_url", "")).strip()
        avg_rating = df_row.get("average_rating", "")
        rating_count = df_row.get("rating_count", None)
        book_id = df_row.get("book_id", "N/A")
        
        # Display image
        if image_url and image_url.lower().startswith(("http://", "https://")):
            try:
                st.image(image_url, width=140)
            except Exception:
                st.image("https://via.placeholder.com/140x210?text=No+Cover", width=140)
        else:
            st.image("https://via.placeholder.com/140x210?text=No+Cover", width=140)
        
        # Display book details below image
        st.markdown(f"**{title}**")
        st.markdown(f"<small>by {authors}</small>", unsafe_allow_html=True)
        
        # Display rating information
        rating_display = []
        if isinstance(avg_rating, (int, float)) and not pd.isna(avg_rating):
            rating_display.append(f"⭐ {avg_rating:.1f}")
        if rating_count and not pd.isna(rating_count):
            rating_display.append(f"📊 {int(rating_count)}")
        
        if rating_display:
            st.markdown(f"<small>{' · '.join(rating_display)}</small>", unsafe_allow_html=True)
        
        # Add book ID in very small text
        st.caption(f"ID: {book_id}")
    
    except Exception as e:
        st.error(f"Error displaying book: {str(e)}")


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Book Recommendation System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .book-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .header-title {
        color: #1f77b4;
        font-weight: 600;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Book Recommendation System")
    st.markdown("Discover your next favorite book with personalized recommendations")
    
    # Load model and data
    model = load()
    books = None
    
    # Load books data
    try:
        if model is not None and "books" in model:
            books = model["books"]
        else:
            books_path = "data/books.csv"
            if os.path.exists(books_path):
                books = pd.read_csv(books_path)
            else:
                st.error(f"Books data not found at {books_path}")
                books = pd.DataFrame(columns=["book_id", "title", "authors", "image_url", "average_rating"])
    except Exception as e:
        st.error(f"Error loading books data: {str(e)}")
        books = pd.DataFrame(columns=["book_id", "title", "authors", "image_url", "average_rating"])
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.selectbox(
        "Recommendation Mode",
        ["By Title", "By Popularity"],
        index=0,
        help="Choose how you want to discover books"
    )
    
    # Model status indicator
    with st.sidebar.expander("Model Status"):
        if model is None:
            st.error("❌ Model not loaded")
            st.info("Train the model using the training script")
        else:
            st.success("✅ Model loaded successfully")
            st.write(f"Users: {len(model.get('user_ids', [])):,}")
            st.write(f"Books: {len(model.get('item_ids', [])):,}")
            st.write(f"Features: {model.get('item_components', np.array([])).shape[0]}")
    
    # Display model warning if missing
    if model is None:
        st.warning("""
        ⚠️ **Model not available**  
        To get personalized recommendations, train the model first:
        ```bash
        python -m src.train \\
          --ratings data/ratings.csv \\
          --books data/books.csv \\
          --out model/svd_model.pkl
        ```
        """)
    
    # Mode-specific content
    if mode == "By Title":
        st.header("🔍 Discover Similar Books")
        st.markdown("Search for a book you like, and we'll recommend similar titles")
        
        # Book search
        search_query = st.sidebar.text_input(
            "🔍 Search by title",
            placeholder="Enter book title...",
            help="Type at least 3 characters to search"
        )
        
        if len(search_query) < 3:
            st.info("🔍 Enter at least 3 characters in the search box to find books")
            return
        
        with st.spinner("Searching books..."):
            matches = lookup_books(books, search_query, max_results=10)
        
        if matches.empty:
            st.warning(f"No books found matching '{search_query}'. Try different keywords.")
            
            # Show popular books as alternatives
            st.subheader("Try these popular books instead:")
            popular_books = get_popular_books(model, books, top_n=6)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(popular_books.iterrows()):
                with cols[idx % 3]:
                    with st.container(border=True):
                        show_book(row)
            return
        
        # Book selection
        selected_title = st.selectbox(
            "Select a book:",
            matches["title"].tolist(),
            index=0,
            help="Choose a book to find similar recommendations"
        )
        
        # Get selected book details
        selected_book = matches[matches["title"] == selected_title].iloc[0]
        
        # Display selected book
        st.subheader("🔖 Your Selected Book")
        col1, col2 = st.columns([1, 4])
        with col1:
            with st.container(border=True):
                show_book(selected_book)
        
        # Show similar books if model is available
        if model is not None:
            st.subheader("📚 Similar Books")
            st.markdown("Books you might enjoy based on your selection")
            
            try:
                with st.spinner("Finding similar books..."):
                    # Ensure book_id is integer
                    book_id = int(selected_book["book_id"])
                    sim_books = find_similar_books_by_book(model, book_id, top_n=12)
                
                # Handle no results
                if not sim_books:
                    st.info("No similar books found. Try selecting a different book.")
                else:
                    # Display similar books in grid
                    cols = st.columns(4)
                    displayed = 0
                    for bid, similarity in sim_books:
                        book_row = books[books["book_id"] == bid]
                        if not book_row.empty:
                            with cols[displayed % 4]:
                                with st.container(border=True):
                                    show_book(book_row.iloc[0])
                                    st.caption(f"Similarity: {similarity:.3f}")
                            displayed += 1
                    
                    if displayed == 0:
                        st.warning("No similar books could be displayed. The model might need retraining.")
            
            except ValueError as ve:
                st.error(f"Invalid book ID format: {str(ve)}")
            except Exception as e:
                st.error(f"Error finding similar books: {str(e)}")
                st.info("This might happen if the book is not in the trained model. Try another book.")
        else:
            st.info("🔍 Similar books require a trained model. Train the model to enable this feature.")
    
    else:  # By Popularity mode
        st.header("🔥 Most Popular Books")
        st.markdown("Discover what other readers are enjoying")
        
        # Number of books to show selector
        top_n = st.sidebar.slider("Number of books to show", 5, 50, 20, 5)
        
        if books is None or books.empty:
            st.error("No book data available. Check your data files and paths.")
            return
        
        with st.spinner(f"Finding top {top_n} popular books..."):
            try:
                popular_books = get_popular_books(model, books, top_n=top_n)
            except Exception as e:
                st.error(f"Error calculating popular books: {str(e)}")
                st.info("Falling back to average rating sort")
                popular_books = books.sort_values("average_rating", ascending=False).head(top_n)
        
        if popular_books.empty:
            st.warning("No popular books found. Try adjusting your filters.")
            return
        
        # Display popular books in grid
        st.markdown(f"### Top {min(top_n, len(popular_books))} Most Popular Books")
        
        # Add metric cards for summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Books Analyzed", len(books))
        with col2:
            avg_rating = popular_books["average_rating"].mean() if "average_rating" in popular_books else 0
            st.metric("Avg Rating (Top Books)", f"{avg_rating:.1f}" if avg_rating else "N/A")
        with col3:
            total_ratings = popular_books["rating_count"].sum() if "rating_count" in popular_books else "N/A"
            st.metric("Total Ratings", f"{total_ratings:,}" if isinstance(total_ratings, (int, float)) else total_ratings)
        
        # Display books in grid
        cols = st.columns(4)
        for idx, (_, row) in enumerate(popular_books.iterrows()):
            with cols[idx % 4]:
                with st.container(border=True):
                    show_book(row)
        
        # Add download button for popular books

if __name__ == "__main__":
    main()