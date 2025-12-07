
import streamlit as st
from kb_core import KB
import tempfile
import os
import time
from pathlib import Path

st.set_page_config(page_title="PDF KB Assistant", layout="wide")


st.sidebar.header("Settings")
index_path = st.sidebar.text_input("FAISS index path", value="kb.index")
meta_path = st.sidebar.text_input("KB metadata path", value="kb_meta.json")
embedding_model = st.sidebar.text_input("Embedding model", value="sentence-transformers/all-mpnet-base-v2")
gen_model = st.sidebar.text_input("Generation model", value="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
device_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)

st.title("PDF Knowledge Base Assistant")
st.markdown("Upload a PDF, build the KB, then ask questions. Answering is sourced from the PDF.")

@st.cache_resource
def get_kb():
    device = 0 if device_gpu else -1
    return KB(index_path=index_path, meta_path=meta_path, embedding_model_name=embedding_model, hf_gen_model=gen_model, device=device)

kb = get_kb()

uploaded = st.file_uploader("Upload a PDF file to build the KB", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
    st.write("Uploaded:", uploaded.name)
    if st.button("Save uploaded PDF and build KB"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        st.info("Saved file to " + tmp_path)
        with st.spinner("Building KB (this may take a while if model downloads)..."):
            try:
                n = kb.build_from_pdf(tmp_path)
                st.success(f"KB built with {n} chunks. Index saved to {index_path}.")
            except Exception as e:
                st.exception("Failed to build KB: " + str(e))

st.markdown("---")
st.header("Ask the KB")
query = st.text_input("Enter your question about the uploaded PDF", "")
top_k = st.slider("Number of retrieved chunks (top_k)", min_value=1, max_value=10, value=5)

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving & generating..."):
                try:
                    answer, sources = kb.answer_query(query, top_k=top_k)
                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Retrieved sources (top results)")
                    for s in sources:
                        st.markdown(f"**chunk_{s['index']}** (score={s['score']:.3f}) — source: {s['meta'].get('source_pdf','-')}")
                        st.write(s['text'][:1000] + (" ..." if len(s['text']) > 1000 else ""))
                        st.write("---")
                except Exception as e:
                    st.exception("Error while answering: " + str(e))

with col2:
    st.markdown("### KB status")
    try:
        if kb.index is None:
            st.write("No KB loaded. Build or upload a PDF.")
        else:
            st.write(f"KB loaded, {len(kb.ids)} chunks.")
    except Exception:
        st.write("KB not initialized.")

st.markdown("---")
st.markdown("### Notes")
st.markdown(
    """
- The DeepSeek model can be large; if you run on CPU, responses may be slow. For production, host the model on an inference server or use a quantized variant.
- If model fails to load with `trust_remote_code=True`, check the model card or use a hosted inference endpoint.
- To expose this Streamlit app publicly, use ngrok (instructions below).
"""
)
