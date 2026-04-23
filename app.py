import streamlit as st
from wiki_loader import load_all_wiki
from predict import predict_with_evidence_list
from retrieval import Retriever

st.set_page_config(
    page_title="Fact Verification System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fact Verification System")
st.markdown("Verify claims using Hybrid Retrieval + DistilBERT")
st.divider()

# --------------------------
# LOAD
# --------------------------
@st.cache_resource
def load_resources():
    wiki = load_all_wiki("data/wiki/")
    retriever = Retriever(wiki)
    return wiki, retriever

wiki, retriever = load_resources()

# --------------------------
# INPUT
# --------------------------
claim = st.text_input("Enter a claim:", placeholder="e.g. Virat Kohli is a cricketer")

col1, col2 = st.columns([1, 5])
with col1:
    verify = st.button("Verify", type="primary")
with col2:
    k = st.slider("Evidence sentences", 1, 10, 5)

# --------------------------
# RUN
# --------------------------
if verify:

    if not claim.strip():
        st.warning("Enter a claim")
    else:
        with st.spinner("Processing..."):

            evidence = retriever.retrieve(claim, k=k)
            evidence_texts = [e["text"] for e in evidence]

            # --------------------------
            # 🔥 FIX 1: evidence quality gate
            # --------------------------
            if len(evidence_texts) == 0:
                result = {
                    "label": "NOT ENOUGH INFO",
                    "scores": {
                        "SUPPORTS": 0.0,
                        "REFUTES": 0.0,
                        "NOT ENOUGH INFO": 100.0
                    }
                }
            else:
                result = predict_with_evidence_list(claim, evidence_texts)

        st.divider()

        label = result["label"]
        scores = result["scores"]

        # --------------------------
        # MAIN RESULT
        # --------------------------
        if label == "SUPPORTS":
            st.success("✅ SUPPORTS")
        elif label == "REFUTES":
            st.error("❌ REFUTES")
        else:
            st.warning("⚠️ NOT ENOUGH INFO")

        # --------------------------
        # SCORES UI
        # --------------------------
        st.subheader("📊 Prediction Scores")

        c1, c2, c3 = st.columns(3)
        c1.metric("SUPPORTS", f"{scores.get('SUPPORTS', 0):.2f}%")
        c2.metric("REFUTES", f"{scores.get('REFUTES', 0):.2f}%")
        c3.metric("NEI", f"{scores.get('NOT ENOUGH INFO', 0):.2f}%")

        # --------------------------
        # EVIDENCE
        # --------------------------
        st.subheader("📄 Evidence")

        if len(evidence) == 0:
            st.info("No strong evidence found.")
        else:
            for i, ev in enumerate(evidence):
                with st.expander(f"{ev['page']} (score {ev['score']:.3f})"):
                    st.write(ev["text"])

        # --------------------------
        # SUMMARY
        # --------------------------
        st.divider()

        st.subheader("🧾 Summary")

        st.markdown(f"""
        | Field | Value |
        |---|---|
        | Claim | {claim} |
        | Verdict | {label} |
        | SUPPORTS | {scores.get('SUPPORTS', 0):.2f}% |
        | REFUTES | {scores.get('REFUTES', 0):.2f}% |
        | NEI | {scores.get('NOT ENOUGH INFO', 0):.2f}% |
        """)