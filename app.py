import streamlit as st
from src.classifier import TextClassifier

st.set_page_config(page_title="Text Moderation", layout="centered")
st.title("Text Moderation System")


@st.cache_resource
def load_model():
    return TextClassifier()


clf = load_model()

st.subheader("Single Analysis")
text = st.text_area("Enter text:", height=100)

if st.button("Analyze"):
    if text.strip():
        detail = clf.get_detail(text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", detail["label"])
        col2.metric("Confidence", f"{detail['confidence'] * 100:.1f}%")
        if detail["allowed"]:
            col3.metric("Result", "ALLOW")
            st.success("PASSED - Content is appropriate")
        else:
            col3.metric("Result", "BLOCK")
            st.error("BLOCKED - Content is not appropriate")

st.divider()

st.subheader("Batch Analysis")
batch_text = st.text_area("Enter one text per line:", height=150, key="batch")

if st.button("Analyze Batch"):
    lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
    if lines:
        results = clf.predict_batch(lines)
        for text, (label, conf) in zip(lines, results):
            allowed = label == "product" and conf >= 0.5
            status = "ALLOW" if allowed else "BLOCK"
            if allowed:
                st.success(f"{text} | {label} | {conf * 100:.1f}% | {status}")
            else:
                st.error(f"{text} | {label} | {conf * 100:.1f}% | {status}")
