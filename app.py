# =====================================================
# E-COMMERCE RAG RECOMMENDER - FULL PIPELINE
# Day 7 Capstone: Production-ready API deployment
# =====================================================

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# =====================================================
# 1. DATA GENERATION (Synthetic E-commerce Dataset)
# =====================================================
@dataclass
class Product:
    id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    rating: float
    image_url: str

def generate_ecommerce_data(num_products: int = 10000) -> List[Product]:
    """
    Generate realistic Indian e-commerce dataset (shoes, apparel, electronics)
    Matches your luxury shoes business interests
    """
    categories = ['Footwear', 'Apparel', 'Electronics', 'Accessories']
    brands = ['Nike', 'Adidas', 'Puma', 'Sparx', 'Bata', 'Samsung', 'Apple', 'Boat']
    
    products = []
    np.random.seed(42)
    
    for i in range(num_products):
        cat = np.random.choice(categories)
        brand = np.random.choice(brands)
        
        if cat == 'Footwear':
            title = f"{brand} {np.random.choice(['Air Max', 'Ultraboost', 'Running', 'Casual', 'Formal']) } Shoes"
            desc = f"Premium {brand} {title.lower()}. Superior cushioning, breathable mesh upper. Perfect for daily wear, gym, running. Available in multiple sizes."
        elif cat == 'Apparel':
            title = f"{brand} {np.random.choice(['T-Shirt', 'Hoodie', 'Trackpants']) }"
            desc = f"100% cotton {brand} {title.lower()}. Regular fit, pre-shrunk fabric. Ideal for casual outings and sports."
        else:
            title = f"{brand} {np.random.choice(['Smartphone', 'Earphones', 'Smartwatch']) }"
            desc = f"Latest {brand} {title.lower()}. Flagship performance, long battery life, premium build quality."
        
        product = Product(
            id=f"prod_{i+1}",
            title=title,
            description=desc,
            category=cat,
            brand=brand,
            price=np.random.uniform(500, 25000, 1)[0],
            rating=round(np.random.uniform(3.5, 5.0, 1)[0], 1),
            image_url=f"https://via.placeholder.com/300x300/FF6B6B/FFFFFF?text={title[:10]}"
        )
        products.append(product)
    
    return products

# =====================================================
# 2. TEXT CHUNKING (Product descriptions → searchable chunks)
# =====================================================
def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """
    Split long product descriptions into overlapping 512-token chunks
    Overlap ensures context preservation across chunks
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length//2):  # 50% overlap
        chunk = ' '.join(words[i:i+max_length])
        if len(chunk) > 20:  # Skip tiny chunks
            chunks.append(chunk)
    return chunks

# =====================================================
# 3. EMBEDDING MODEL (Semantic search backbone)
# =====================================================
class EcommerceEmbedder:
    def __init__(self):
        """Fast, lightweight embedding model optimized for product search"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.eval()
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Batch embedding for 10K+ product chunks"""
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding"""
        return self.model.encode([query]).astype('float32')

# =====================================================
# 4. FAISS VECTOR STORE (10K products → <1s search)
# =====================================================
class EcommerceVectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.embedder = EcommerceEmbedder()
    
    def build_from_products(self, products: List[Product]):
        """Full pipeline: chunk → embed → index"""
        print("🔄 Chunking product descriptions...")
        all_chunks = []
        all_metadata = []
        
        for prod in products:
            chunks = chunk_text(prod.description)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'product_id': prod.id,
                    'title': prod.title,
                    'chunk_id': i,
                    'price': prod.price,
                    'category': prod.category,
                    'brand': prod.brand,
                    'rating': prod.rating,
                    'full_desc': prod.description
                })
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        print(f"📊 Created {len(all_chunks)} chunks from {len(products)} products")
        
        print("🔄 Embedding chunks...")
        embeddings = self.embedder.embed_documents(all_chunks)
        
        print("🔄 Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Cosine similarity
        faiss.normalize_L2(embeddings)  # L2 normalize for cosine
        self.index.add(embeddings)
        
        print(f"✅ FAISS index ready: {self.index.ntotal} vectors")
    
    def semantic_search(self, query: str, k: int = 20) -> List[Dict]:
        """Retrieve TOP-K most similar product chunks"""
        query_emb = self.embedder.embed_query(query)
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, k)
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        return results

# =====================================================
# 5. RERANKER (Cross-encoder for precise relevance)
# =====================================================
class Reranker:
    def __init__(self):
        """Cross-encoder reranks TOP-20 → TOP-10 with high precision"""
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model.eval()
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank candidates by query-chunk relevance score"""
        pairs = [[query, cand['metadata']['title'] + " " + cand['metadata']['full_desc']] 
                for cand in candidates]
        
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().tolist()
        
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = score
        
        # Sort by rerank score (higher = better)
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

# =====================================================
# 6. RECOMMENDATION ENGINE (Filters + LLM reasoning)
# =====================================================
class EcommerceRecommender:
    def __init__(self):
        self.vectorstore = EcommerceVectorStore()
        self.reranker = Reranker()
    
    def recommend(self, query: str, max_price: float = None, category: str = None, top_k: int = 5) -> Dict:
        """
        Full recommendation pipeline:
        1. Semantic search TOP-20
        2. Rerank TOP-10  
        3. Filter constraints
        4. LLM reasoning → final TOP-5
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Step 1: Semantic retrieval
        candidates = self.vectorstore.semantic_search(query, k=20)
        
        # Step 2: Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=10)
        
        # Step 3: Apply filters
        filtered = self._apply_filters(reranked, max_price, category)
        
        # Step 4: LLM reasoning (OpenAI GPT-4o-mini)
        final_recs = self._llm_reasoning(query, filtered, top_k)
        
        return {
            "query": query,
            "recommendations": final_recs,
            "retrieved_count": len(candidates),
            "filtered_count": len(filtered)
        }
    
    def _apply_filters(self, candidates: List[Dict], max_price: float, category: str) -> List[Dict]:
        """Hard filters: price, category"""
        filtered = []
        for cand in candidates:
            meta = cand['metadata']
            price_ok = max_price is None or meta['price'] <= max_price
            cat_ok = category is None or meta['category'].lower() == category.lower()
            
            if price_ok and cat_ok:
                filtered.append(cand)
        return filtered
    
    def _llm_reasoning(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Use LLM to select final recommendations with reasoning"""
        context = "\n".join([
            f"Product {i+1}: {c['metadata']['title']} (₹{c['metadata']['price']:.0f}, "
            f"{c['metadata']['brand']}, Score: {c['rerank_score']:.3f})"
            for i, c in enumerate(candidates[:8])  # TOP-8 context
        ])
        
        prompt = f"""
        E-commerce recommendation task.
        
        User query: "{query}"
        Available products:
        {context}
        
        Select TOP-{top_k} products. Return ONLY JSON:
        {{
            "recommendations": [
                {{"rank": 1, "product_id": "...", "title": "...", "price": ..., "reason": "..."}},
                ...
            ],
            "reasoning": "Why these products match the query best"
        }}
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result['recommendations']

# =====================================================
# 7. FASTAPI ENDPOINT (Production API)
# =====================================================
app = FastAPI(title="E-commerce RAG Recommender API")

class QueryRequest(BaseModel):
    query: str
    max_price: float = None
    category: str = None
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    recommendations: List[Dict[str, Any]]
    retrieved_count: int
    filtered_count: int

recommender = None  # Global instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize recommender on startup"""
    global recommender
    print("🚀 Initializing E-commerce RAG Recommender...")
    
    # Generate data
    products = generate_ecommerce_data(5000)  # 5K products
    recommender = EcommerceRecommender()
    recommender.vectorstore.build_from_products(products)
    
    print("✅ Recommender ready!")
    yield
    print("👋 Shutting down...")

app.router.lifespan_context = lifespan

@app.post("/recommend", response_model=QueryResponse)
async def recommend(request: QueryRequest):
    """Main recommendation endpoint"""
    result = recommender.recommend(
        query=request.query,
        max_price=request.max_price,
        category=request.category,
        top_k=request.top_k
    )
    return result

@app.get("/health")
async def health():
    return {"status": "healthy", "products_indexed": recommender.vectorstore.index.ntotal}

# =====================================================
# 8. STREAMLIT UI (Demo interface)
# =====================================================
def streamlit_ui():
    st.title("🛍️ E-commerce RAG Recommender")
    st.markdown("**Day 7 Capstone**: Semantic product search + LLM reasoning")
    
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input("Search products", "luxury running shoes under 5000")
        max_price = st.number_input("Max price (₹)", value=5000.0)
    
    with col2:
        category = st.selectbox("Category", ["All", "Footwear", "Apparel", "Electronics"])
        top_k = st.slider("Recommendations", 1, 10, 5)
    
    if st.button("🔍 Get Recommendations", type="primary"):
        category_filter = category if category != "All" else None
        
        with st.spinner("Searching 5K products..."):
            result = recommender.recommend(
                query=query,
                max_price=max_price,
                category=category_filter,
                top_k=top_k
            )
        
        st.success(f"Found {result['retrieved_count']} similar products")
        
        for rec in result['recommendations']:
            with st.container():
                st.markdown(f"**#{rec['rank']}** {rec['title']}")
                st.caption(f"₹{rec['price']:.0f} | Reason: {rec['reason']}")
                st.divider()

# =====================================================
# 9. MAIN EXECUTION (Run modes)
# =====================================================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "streamlit"
    
    if mode == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif mode == "streamlit":
        streamlit_ui()
