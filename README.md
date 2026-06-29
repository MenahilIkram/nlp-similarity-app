# ✨ NeuroSim — Semantic Similarity Explorer

A glassmorphism-styled Streamlit web app that uses a **free pretrained NLP model** to compute and visualize text/sentence similarity — built for the **Shifa Tameer-e-Millat University, Natural Language Processing Lab** quiz.

## 📌 App Purpose
This app lets a user type a query sentence plus several candidate sentences. It loads a **free pretrained sentence embedding model directly** (no training, no manual preprocessing, no fine-tuning) and computes cosine similarity between the query and each candidate. Results are shown as exact similarity scores, supported by three interactive graphs, plus short notes following **Paul's Critical Thinking Standards**.

## 🤖 Pretrained Model Used
- **Model:** `all-MiniLM-L6-v2`
- **Library:** [sentence-transformers](https://www.sbert.net/)
- **Type:** Free, open-source pretrained sentence embedding model (no API key, no payment, no training performed)

## 🚀 Features
1. Text input box for a query + multiple comparison lines.
2. Pretrained model (`all-MiniLM-L6-v2`) loaded directly via `sentence-transformers`.
3. Exact cosine similarity scores for top matching results.
4. Three supporting graphs:
   - **Bar Chart** — Top-N similar items with exact scores.
   - **Heatmap** — Pairwise similarity matrix across all entered items.
   - **3D PCA Embedding Plot** — Rotating 3D scatter plot showing how related texts cluster in embedding space.
5. Critical thinking notes (Clarity, Accuracy, Precision, Relevance, Logic, Significance, Fairness) generated dynamically from the actual results.

## 🛠️ Tech Stack
- Streamlit (UI + deployment)
- sentence-transformers (pretrained embeddings)
- Plotly (interactive 2D/3D graphs)
- scikit-learn (PCA dimensionality reduction)
- NumPy

## ▶️ Run Locally
```bash
git clone <this-repo-url>
cd nlp-similarity-app
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployed App
**Streamlit Link:** _https://nlp-similarity-app-l4zjeheiuqtct2zycamsje.streamlit.app/

## 📷 Screenshots


```
home page
<img width="1899" height="726" alt="image" src="https://github.com/user-attachments/assets/af068453-bede-4392-a4d4-ee8c0a877ddd" />

bar_chart
<img width="1322" height="740" alt="image" src="https://github.com/user-attachments/assets/0c3511de-e330-48ca-a17f-d7bc5bc826a5" />
heatmap
<img width="1344" height="702" alt="image" src="https://github.com/user-attachments/assets/61e1b456-b772-4d7e-be13-20c12bed67de" />

3d_plot
<img width="1010" height="629" alt="image" src="https://github.com/user-attachments/assets/4042e84c-ece6-489d-bebc-e1e839110bb0" />

```

## ⚠️ Note on Rules
- ✅ No model training performed.
- ✅ No manual preprocessing (no stopword removal, stemming, lemmatization, manual tokenization).
- ✅ Only a free pretrained model is used (`all-MiniLM-L6-v2`).
- ✅ No paid APIs or paid models used.

## 📚 Critical Thinking Standards Applied
| Standard | Implementation |
|---|---|
| Clarity | Explains input query and what similarity scores mean |
| Accuracy | States model name (`all-MiniLM-L6-v2`), avoids overclaiming |
| Precision | Shows exact decimal similarity scores |
| Relevance | All 3 graphs directly visualize the computed similarity results |
| Logic | Explains why top result has the highest score |
| Significance | Highlights the score gap between best and worst match |
| Fairness | States MiniLM's limitation as a general-purpose model |

---
