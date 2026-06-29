# 🔍 NLP Text Similarity Explorer

**NLP Lab Quiz | Shifa Tameer-e-Millat University, Islamabad**

## 📌 App Purpose
This Streamlit web app uses a **free pretrained NLP model** to compute and visualize semantic text similarity between user-provided sentences. No preprocessing, training, or paid APIs are used.

## 🤖 Pretrained Model
- **Model Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Source:** HuggingFace (Free)
- **Task:** Sentence embeddings + cosine similarity

## 🚀 Streamlit App Link
> **[Click here to open the live app](YOUR_STREAMLIT_LINK_HERE)**

## ✨ Features
- Enter 2–8 sentences/words
- Enter a custom query to find its top matching sentence
- Similarity scores computed via cosine similarity on 384-dim embeddings
- Paul's Critical Thinking Standards applied to every result

## 📊 Graphs Included
| Graph | Purpose |
|---|---|
| Bar Chart | Top similar sentences ranked by score |
| Heatmap | Pairwise cosine similarity between all inputs |
| 2D PCA Plot | Embedding space visualization |

## 🔧 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure
```
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 📸 Screenshots
*(Add screenshots here after deployment)*

## ⚠️ Rules Followed
- ✅ No manual tokenization or preprocessing
- ✅ No model training
- ✅ Free pretrained model only
- ✅ Deployed on Streamlit Community Cloud
