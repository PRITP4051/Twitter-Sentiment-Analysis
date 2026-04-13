import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert_model")
    tokenizer = BertTokenizer.from_pretrained("bert_model")
    return model, tokenizer

model, tokenizer = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔍 Prediction", "📊 Dataset", "📈 Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using BERT")

# =====================================================
# 🔍 PAGE 1: PREDICTION
# =====================================================
if page == "🔍 Prediction":

    st.title("🤖 AI Sentiment Analyzer")
    st.markdown("### Analyze sentiment using BERT (Transformer Model)")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    text = st.text_area("✍️ Enter your text here:", height=150)

    if st.button("🚀 Predict Sentiment"):
        if text.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            pred = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

            st.markdown("---")

            # Sentiment display
            if pred == 1:
                st.success("😊 Positive Sentiment")
            elif pred == 0:
                st.error("😡 Negative Sentiment")
            else:
                st.warning("😐 Neutral Sentiment")

            # Confidence Score
            st.markdown("### 🎯 Confidence Score")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# 📊 PAGE 2: DATASET
# =====================================================
elif page == "📊 Dataset":

    st.title("📊 Dataset Viewer")

    df = pd.read_csv("labeled_data.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])

    st.markdown("---")

    st.dataframe(df, use_container_width=True)

# =====================================================
# 📈 PAGE 3: MODEL COMPARISON
# =====================================================
elif page == "📈 Model Comparison":

    st.title("📈 Model Performance Comparison")

    df = pd.read_csv("final_model_comparison.csv")

    st.markdown("### 📊 Performance Table")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Test Accuracy")
        st.bar_chart(df.set_index("model")[["test_accuracy"]])

    with col2:
        st.subheader("🎯 F1 Score")
        st.bar_chart(df.set_index("model")[["f1_score"]])

    st.markdown("---")

    best_model = df.loc[df['test_accuracy'].idxmax()]

    st.success(
        f"🏆 Best Model: {best_model['model']} "
        f"(Accuracy: {best_model['test_accuracy']:.2f})"
    )