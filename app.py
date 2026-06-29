import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NLP Text Similarity Explorer", page_icon="🔍", layout="wide")

st.title("🔍 NLP Text Similarity Explorer")
st.markdown("**Pretrained Model:** `sentence-transformers/all-MiniLM-L6-v2` (Free, HuggingFace)")
st.markdown("---")

# ─── Load Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()
st.success("✅ Model loaded: all-MiniLM-L6-v2")

# ─── Input Section ─────────────────────────────────────────────────────────────
st.header("📝 Input")
st.markdown("Enter **2 to 8** sentences or words (one per line). The app will compute pairwise similarity.")

default_inputs = "The cat sat on the mat\nThe dog rested on the rug\nArtificial intelligence is transforming technology\nMachine learning powers modern AI systems\nThe weather is sunny today\nI love eating pizza"

user_input = st.text_area("Enter your sentences/words (one per line):", value=default_inputs, height=160)
query = st.text_input("🔎 Enter a query sentence to find its top matches:", value="deep learning is a subset of AI")

run = st.button("▶ Compute Similarity", type="primary")

if run:
    lines = [line.strip() for line in user_input.strip().split("\n") if line.strip()]

    if len(lines) < 2:
        st.error("Please enter at least 2 sentences.")
        st.stop()
    if len(lines) > 8:
        lines = lines[:8]
        st.warning("Trimmed to first 8 sentences for readability.")

    # ─── Compute Embeddings ──────────────────────────────────────────────────
    with st.spinner("Computing embeddings..."):
        embeddings = model.encode(lines, convert_to_tensor=True)
        cosine_matrix = util.cos_sim(embeddings, embeddings).numpy()

    # ─── Query Similarity ────────────────────────────────────────────────────
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_scores = util.cos_sim(query_embedding, embeddings).numpy().flatten()
    sorted_idx = np.argsort(query_scores)[::-1]

    st.markdown("---")
    st.header("📊 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top Matches for Query")
        st.markdown(f"> *\"{query}\"*")
        for rank, i in enumerate(sorted_idx):
            score = query_scores[i]
            st.markdown(f"**{rank+1}.** `{score:.4f}` — {lines[i]}")

    with col2:
        st.subheader("Similarity Score Table")
        import pandas as pd
        df = pd.DataFrame(cosine_matrix, index=lines, columns=lines).round(4)
        st.dataframe(df, use_container_width=True)

    # ─── GRAPH 1: Bar Chart ──────────────────────────────────────────────────
    st.markdown("---")
    st.header("📈 Graph 1 — Bar Chart: Query Similarity Scores")

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    colors = ["#2ecc71" if i == sorted_idx[0] else "#3498db" for i in range(len(lines))]
    short_labels = [l[:35] + "..." if len(l) > 35 else l for l in lines]
    ax1.barh(short_labels, query_scores, color=colors)
    ax1.set_xlabel("Cosine Similarity Score")
    ax1.set_title(f"Similarity to Query: \"{query[:50]}\"")
    ax1.set_xlim(0, 1)
    for i, v in enumerate(query_scores):
        ax1.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
    ax1.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig1)

    # ─── GRAPH 2: Heatmap ────────────────────────────────────────────────────
    st.header("🌡️ Graph 2 — Heatmap: Pairwise Similarity")

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        cosine_matrix,
        annot=True, fmt=".2f",
        xticklabels=short_labels, yticklabels=short_labels,
        cmap="YlOrRd", linewidths=0.5, ax=ax2,
        vmin=0, vmax=1
    )
    ax2.set_title("Pairwise Cosine Similarity Heatmap")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

    # ─── GRAPH 3: 2D PCA Plot ────────────────────────────────────────────────
    st.header("🗺️ Graph 3 — 2D Embedding Plot (PCA)")

    emb_np = embeddings.numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(emb_np)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    scatter = ax3.scatter(reduced[:, 0], reduced[:, 1], c=np.arange(len(lines)), cmap="tab10", s=120, zorder=3)
    for i, label in enumerate(short_labels):
        ax3.annotate(label, (reduced[i, 0], reduced[i, 1]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax3.set_title("2D PCA Projection of Sentence Embeddings")
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    ax3.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig3)

    # ─── Paul's Critical Thinking Notes ─────────────────────────────────────
    st.markdown("---")
    st.header("🧠 Critical Thinking Analysis (Paul's Standards)")

    top_match = lines[sorted_idx[0]]
    top_score = query_scores[sorted_idx[0]]
    second_match = lines[sorted_idx[1]] if len(sorted_idx) > 1 else ""
    second_score = query_scores[sorted_idx[1]] if len(sorted_idx) > 1 else 0

    ct_data = {
        "🔵 Clarity": f"The user provided {len(lines)} sentences. The app computed cosine similarity between each sentence and the query using dense vector embeddings. The output shows how semantically close each sentence is to the query.",
        "✅ Accuracy": f"The pretrained model used is **sentence-transformers/all-MiniLM-L6-v2** from HuggingFace. No training, fine-tuning, or preprocessing was performed. All similarity scores are direct model outputs.",
        "🎯 Precision": f"The top match is: *\"{top_match}\"* with an exact cosine similarity score of **{top_score:.4f}**. The second match scored **{second_score:.4f}**, a difference of {abs(top_score - second_score):.4f}.",
        "🔗 Relevance": "All three graphs directly support the results — the bar chart shows ranked scores, the heatmap shows sentence-to-sentence similarity, and PCA confirms cluster structure in embedding space.",
        "⚙️ Logic": f"The highest-scoring sentence is semantically closest to the query because MiniLM encodes meaning into 384-dimensional vectors. Sentences sharing topics, vocabulary, or context have higher cosine similarity.",
        "⭐ Significance": f"The most important result is that *\"{top_match[:60]}\"* achieves the highest similarity of **{top_score:.4f}**, indicating it is the most semantically relevant sentence to the query in this input set.",
        "⚠️ Fairness (Limitation)": "MiniLM is a general-purpose model trained on English data. It may produce lower-quality embeddings for domain-specific text (medical, legal), non-English input, or very short single words where context is minimal.",
    }

    for standard, explanation in ct_data.items():
        with st.expander(standard):
            st.markdown(explanation)

    st.markdown("---")
    st.caption("NLP Lab Quiz | Shifa Tameer-e-Millat University | Model: all-MiniLM-L6-v2 (HuggingFace, Free)")
