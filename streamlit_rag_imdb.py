import streamlit as streamlit
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os

# ============================================================
# 1. Configure page and load components
# ============================================================

streamlit.set_page_config(
    page_title="IMDB RAG Assistant",
    page_icon="🎬",
    layout="wide"
)

streamlit.title("🎬 IMDB Movie Review RAG Assistant")
streamlit.markdown("Ask questions about movies using real IMDB reviews!")

# Sidebar for configuration
streamlit.sidebar.header("Configuration")
openai_key = streamlit.sidebar.text_input("OpenAI API Key", type="password")
embedding_model = streamlit.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2"]
)

if "faiss_store" not in streamlit.session_state:
    with streamlit.spinner("Loading FAISS vector store..."):
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )

        # Load FAISS index from local store
        streamlit.session_state.faiss_store = FAISS.load_local(
            "imdb_faiss_store",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        streamlit.session_state.retriever = streamlit.session_state.faiss_store.as_retriever(search_kwargs={"k": 4})

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    streamlit.session_state.llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1
    )
    streamlit.sidebar.success("✅ OpenAI configured!")
else:
    streamlit.sidebar.warning("⚠️ Enter OpenAI API key to use LLM")

# ============================================================
# 2. Main query interface
# ============================================================

# Query input
query = streamlit.text_area(
    "Ask a question about IMDB movie reviews:",
    height=100,
    placeholder="e.g., What do reviewers think about the acting in this movie?"
)

if streamlit.button("🔍 Search IMDB Reviews & Generate Answer", type="primary") and query:
    if "llm" not in streamlit.session_state:
        streamlit.error("Please enter your OpenAI API key in the sidebar!")
    else:
        with streamlit.spinner("Retrieving relevant reviews and generating answer..."):
            # Retrieve relevant documents
            docs = streamlit.session_state.retriever.invoke(query)

            # Build context from top 4 docs
            context_parts = []
            for i, doc in enumerate(docs[:4]):
                snippet = doc.page_content[:400].replace("\n", " ")
                context_parts.append(f"**Review {i + 1}:** {snippet}")
            context = "\n\n".join(context_parts)

            # Prompt template
            prompt_template = ChatPromptTemplate.from_template(
                """
You are a helpful assistant answering questions about movie reviews from IMDB.
Use ONLY the provided IMDB review excerpts as context. Be specific about what reviewers say.

IMDB Reviews Context:
{context}

Question: {question}

Answer in 3-5 sentences, citing specific review quotes when possible.
"""
            )

            # Generate answer
            messages = prompt_template.format_messages(
                context=context,
                question=query
            )
            response = streamlit.session_state.llm.invoke(messages)
            answer = response.content

            # ============================================================
            # 3. Display results
            # ============================================================

            col1, col2 = streamlit.columns([2, 1])

            with col1:
                streamlit.markdown("## 🤖 **Answer**")
                streamlit.write(answer)

            with col2:
                streamlit.markdown("## 📊 **Retrieved Reviews**")
                for i, doc in enumerate(docs[:3]):
                    with streamlit.expander(f"Review {i + 1} (Label: {doc.metadata.get('label', 'N/A')})"):
                        streamlit.write(doc.page_content[:500] + "...")

# ============================================================
# 4. Example queries
# ============================================================

with streamlit.expander("💡 Example Questions"):
    streamlit.markdown("""
    - "What do reviewers think about the acting?"
    - "Which movies have great cinematography?"
    - "Are there any complaints about the plot?"
    - "What makes this movie boring according to reviews?"
    """)

streamlit.markdown("---")
streamlit.markdown("**Powered by:** FAISS + Sentence Transformers + OpenAI + LangChain + Streamlit")
