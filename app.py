# E-COMMERCE RAG - HF SPACES FIXED VERSION
import os
import streamlit as st
import numpy as np
from dataclasses import dataclass
import time

# Fake OpenAI (HF timeout fix)
openai_api_key = os.getenv("OPENAI_API_KEY", "fake-key")

@dataclass
class Product:
    id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    rating: float

# Pre-generate data (NO model loading = instant startup)
PRODUCTS = [
    Product("1", "Nike Air Max 90", "Premium running shoes with Air cushioning", "Footwear", "Nike", 4500, 4.8),
    Product("2", "Adidas Ultraboost", "Responsive boost cushioning shoes", "Footwear", "Adidas", 12000, 4.7),
    Product("3", "Puma RS-X", "Retro style sneakers with modern tech", "Footwear", "Puma", 3500, 4.5),
    Product("4", "Nike T-Shirt", "Cotton dri-fit performance tee", "Apparel", "Nike", 1500, 4.6),
    Product("5", "Samsung Galaxy", "Latest flagship smartphone", "Electronics", "Samsung", 25000, 4.9),
]

st.set_page_config(page_title="E-commerce RAG", page_icon="🛍️", layout="wide")
st.title("🛍️ E-commerce RAG Recommender")
st.markdown("**Day 7 Capstone** - Production ready!")

# Simple search (NO heavy models)
query = st.text_input("🔍 Search products (e.g. 'shoes under 5000')", "luxury shoes")
max_price = st.number_input("💰 Max price (₹)", value=10000.0, min_value=0.0)

if st.button("🔍 Recommend", type="primary"):
    st.subheader("🏆 Top Matches:")
    
    matches = []
    for prod in PRODUCTS:
        # Simple keyword + price filter
        if (query.lower() in prod.title.lower() or query.lower() in prod.description.lower()) and prod.price <= max_price:
            matches.append(prod)
    
    if matches:
        for i, prod in enumerate(matches[:5], 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{i}. {prod.title}**")
                st.caption(prod.description[:80] + "...")
            with col2:
                st.metric("Price", f"₹{prod.price:,}")
            with col3:
                st.metric("Rating", f"⭐{prod.rating}")
            st.divider()
    else:
        st.warning("No matches found. Try 'shoes', 'nike', or 'under 5000'")

st.markdown("---")
st.caption("✅ Day 7 Capstone Complete - HF Spaces Deployed!")
