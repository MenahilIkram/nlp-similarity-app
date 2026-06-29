import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroSim | Semantic Similarity Explorer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLASSMORPHISM DARK THEME + ANIMATED BACKGROUND
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated gradient + floating particles background */
.stApp {
    background: radial-gradient(circle at 20% 20%, #1b1037 0%, #0a0a18 45%, #05050d 100%);
    overflow-x: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(2px 2px at 20% 30%, rgba(168,85,247,0.6), transparent),
        radial-gradient(2px 2px at 70% 10%, rgba(56,189,248,0.6), transparent),
        radial-gradient(2px 2px at 40% 70%, rgba(236,72,153,0.5), transparent),
        radial-gradient(2px 2px at 90% 60%, rgba(168,85,247,0.5), transparent),
        radial-gradient(2px 2px at 10% 90%, rgba(56,189,248,0.5), transparent),
        radial-gradient(2px 2px at 60% 85%, rgba(236,72,153,0.4), transparent);
    background-size: 200% 200%;
    animation: drift 30s linear infinite;
    z-index: 0;
    pointer-events: none;
}

@keyframes drift {
    0%   { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}

/* Glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 22px 26px;
    margin-bottom: 18px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px 0 rgba(168,85,247,0.25);
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(90deg, #a855f7, #38bdf8, #ec4899);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-align: center;
    margin-bottom: 0;
    animation: glow 4s ease-in-out infinite alternate;
}
@keyframes glow {
    from { filter: drop-shadow(0 0 8px rgba(168,85,247,0.4)); }
    to   { filter: drop-shadow(0 0 22px rgba(56,189,248,0.6)); }
}
.hero-sub {
    text-align: center;
    color: #c4b5fd;
    font-size: 1.05rem;
    margin-top: 4px;
    margin-bottom: 28px;
    letter-spacing: 0.5px;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(168,85,247,0.25), rgba(56,189,248,0.25));
    border: 1px solid rgba(255,255,255,0.18);
    color: #e9d5ff;
    font-size: 0.8rem;
    margin-right: 6px;
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: #f5f3ff;
    border-left: 4px solid #a855f7;
    padding-left: 12px;
    margin-top: 10px;
    margin-bottom: 14px;
}

.cts-row {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 10px;
    border-left: 3px solid #38bdf8;
}
.cts-label {
    color: #38bdf8;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(10, 8, 26, 0.85);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

textarea, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

div.stButton > button {
    background: linear-gradient(90deg, #a855f7, #38bdf8);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 10px 26px;
    font-weight: 600;
    transition: transform 0.2s ease;
}
div.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 0 18px rgba(168,85,247,0.5);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">✨ NeuroSim</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">A Semantic Similarity Explorer powered by a free pretrained Sentence-Transformer model</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; margin-bottom:24px;">'
    '<span class="badge">🤖 all-MiniLM-L6-v2</span>'
    '<span class="badge">🧠 Sentence-Transformers</span>'
    '<span class="badge">📊 3D + 2D Visualizations</span>'
    '<span class="badge">🆓 Free &amp; Pretrained</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LOAD PRETRAINED MODEL (cached, no training, no fine-tuning)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading free pretrained model (all-MiniLM-L6-v2)...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("Enter one item **per line**. The first line is treated as the *query*; remaining lines are compared against it.")
    sample = st.checkbox("Load sample data", value=True)
    top_n = st.slider("Top-N results to show", 3, 10, 5)
    st.markdown("---")
    st.markdown("**Model:** `all-MiniLM-L6-v2`")
    st.markdown("**Library:** `sentence-transformers`")
    st.markdown("**Type:** Free pretrained embedding model (no training used)")

# ----------------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">📝 Step 1 — Enter Text</div>', unsafe_allow_html=True)

default_text = """I love natural language processing
Machine learning is fascinating
I enjoy studying artificial intelligence
The weather is nice today
Deep learning models are powerful
I went to the market to buy vegetables
Neural networks can understand language"""

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    user_text = st.text_area(
        "Enter your query (first line) and comparison items (one per line):",
        value=default_text if sample else "",
        height=180,
        placeholder="Type the query sentence on the first line, then one item per line below...",
    )
    run = st.button("🚀 Run Similarity Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------------------------
if run and user_text.strip():
    lines = [l.strip() for l in user_text.split("\n") if l.strip()]
    if len(lines) < 2:
        st.warning("Please enter at least a query line and one comparison line.")
        st.stop()

    query = lines[0]
    candidates = lines[1:]
    all_items = [query] + candidates

    # Embeddings from the pretrained model directly — no preprocessing applied
    embeddings = model.encode(all_items)

    query_emb = embeddings[0]
    cand_embs = embeddings[1:]

    # Cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = np.array([cosine_sim(query_emb, c) for c in cand_embs])
    order = np.argsort(-sims)
    top_n_eff = min(top_n, len(candidates))
    top_idx = order[:top_n_eff]

    # ---------------------- RESULTS TABLE ----------------------
    st.markdown('<div class="section-header">📌 Step 2 — Similarity Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"**Query:** _{query}_")
    for rank, idx in enumerate(top_idx, start=1):
        st.markdown(f"**{rank}.** {candidates[idx]}  —  similarity score: **{sims[idx]:.4f}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- GRAPH 1: BAR CHART ----------------------
    st.markdown('<div class="section-header">📊 Graph 1 — Top Similar Items (Bar Chart)</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    bar_labels = [candidates[i] for i in top_idx]
    bar_scores = [sims[i] for i in top_idx]
    fig_bar = px.bar(
        x=bar_scores, y=bar_labels, orientation="h",
        color=bar_scores, color_continuous_scale=["#38bdf8", "#a855f7", "#ec4899"],
        labels={"x": "Cosine Similarity Score", "y": ""},
    )
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"), height=420, coloraxis_showscale=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- GRAPH 2: HEATMAP ----------------------
    st.markdown('<div class="section-header">🔥 Graph 2 — Pairwise Similarity (Heatmap)</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    sim_matrix = np.zeros((len(all_items), len(all_items)))
    for i in range(len(all_items)):
        for j in range(len(all_items)):
            sim_matrix[i, j] = cosine_sim(embeddings[i], embeddings[j])

    short_labels = [t if len(t) <= 22 else t[:20] + "…" for t in all_items]
    fig_heat = go.Figure(data=go.Heatmap(
        z=sim_matrix, x=short_labels, y=short_labels,
        colorscale=[[0, "#0a0a18"], [0.5, "#a855f7"], [1, "#38bdf8"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Score"),
    ))
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500, xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- GRAPH 3: 3D EMBEDDING PLOT (PCA) ----------------------
    st.markdown('<div class="section-header">🌌 Graph 3 — 3D Rotating Embedding Space (PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    n_components = min(3, len(all_items))
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    if reduced.shape[1] < 3:
        pad = np.zeros((reduced.shape[0], 3 - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])

    colors = ["#ec4899"] + ["#38bdf8"] * len(candidates)
    sizes = [22] + [14] * len(candidates)

    fig3d = go.Figure(data=[go.Scatter3d(
        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
        mode="markers+text",
        text=short_labels,
        textposition="top center",
        marker=dict(size=sizes, color=colors, opacity=0.9, line=dict(width=1, color="white")),
    )])
    fig3d.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="PC1", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="PC2", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="PC3", backgroundcolor="rgba(0,0,0,0)"),
        ),
        height=620,
        margin=dict(l=0, r=0, t=10, b=0),
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("🖱️ Drag to rotate the 3D embedding space — pink = query, blue = candidates.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- CRITICAL THINKING NOTES ----------------------
    best_idx = top_idx[0]
    worst_idx = order[-1]
    st.markdown('<div class="section-header">🧩 Step 3 — Critical Thinking Notes (Paul\'s Standards)</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    notes = {
        "Clarity": f"The input query was \"{query}\", compared against {len(candidates)} candidate texts. The output is a cosine similarity score (0 to 1) showing how close each candidate's meaning is to the query, based on sentence embeddings.",
        "Accuracy": "Results come from the pretrained `all-MiniLM-L6-v2` Sentence-Transformer model. No training or fine-tuning was performed, so scores reflect this model's general-purpose semantic understanding only, not domain-specific accuracy.",
        "Precision": f"Exact scores are reported, e.g. the top match \"{candidates[best_idx]}\" scored **{sims[best_idx]:.4f}**, instead of vague labels like 'high' or 'low' similarity.",
        "Relevance": "The bar chart shows ranked top matches, the heatmap shows all pairwise relationships, and the 3D PCA plot visually places semantically similar texts closer together — all three directly support the similarity scores reported above.",
        "Logic": f"\"{candidates[best_idx]}\" ranks highest because its sentence embedding lies closest in vector space to the query embedding, meaning the model judged their overall meaning to be most alike — visible in the 3D plot as the nearest point to the query.",
        "Significance": f"The most important finding is that the score gap between the top match ({sims[best_idx]:.4f}) and the lowest match ({sims[worst_idx]:.4f}) shows the model can meaningfully separate related from unrelated text.",
        "Fairness": "Limitation: MiniLM is a general-purpose model trained on broad web/text corpora — it may miss nuanced domain-specific meaning, sarcasm, or context-dependent phrasing, so scores should be interpreted as approximate semantic closeness, not ground truth.",
    }
    for std, txt in notes.items():
        st.markdown(f'<div class="cts-row"><span class="cts-label">{std}:</span> {txt}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Enter your text in the box above and click **Run Similarity Analysis** to see results, graphs, and critical thinking notes.")

st.markdown(
    '<div style="text-align:center; margin-top:40px; color:#7c7c9c; font-size:0.85rem;">'
    'Built for Shifa Tameer-e-Millat University · NLP Lab Quiz · Free Pretrained Model Demo'
    '</div>',
    unsafe_allow_html=True,
)
