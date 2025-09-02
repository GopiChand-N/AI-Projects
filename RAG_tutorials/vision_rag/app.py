import os, requests, numpy as np, tqdm
from PIL import Image
import streamlit as st
from embeddings import ClipEmbedder
from retrieval import find_top
from pdf_utils import process_pdf_file
from qa import answer_with_gpt4o

st.set_page_config(layout="wide", page_title="Vision RAG (CLIP + GPT-4o)")
st.title("Vision RAG (CLIP + GPT-4o)")

with st.sidebar:
    st.header("API Key")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    st.markdown("---")
    st.caption("CLIP runs locally. GPT-4o uses your OpenAI key.")

@st.cache_resource(show_spinner=False)
def load_embedder():
    return ClipEmbedder()

embedder = load_embedder()

if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = None

with st.expander("About"):
    st.write("Embeds images and PDF pages with CLIP; retrieves the best match for your question; GPT-4o answers using that image.")

SAMPLE_IMAGES = {
    "tesla.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
    "netflix.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
    "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    "google.png": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
    "accenture.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
    "tecent.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png"
}

@st.cache_data(ttl=3600, show_spinner=False)
def download_and_embed_samples(_embedder: ClipEmbedder):
    os.makedirs("img", exist_ok=True)
    paths, embs = [], []
    for name, url in tqdm.tqdm(SAMPLE_IMAGES.items(), desc="samples"):
        p = os.path.join("img", name)
        print("path", p)
        if not os.path.exists(p):
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(p, "wb") as f:
                f.write(r.content)
        img = Image.open(p)
        emb = _embedder.image_embedding(img)
        paths.append(p)
        embs.append(emb)
    return paths, (np.vstack(embs) if embs else None)

st.subheader("Load sample images")
if st.button("Load samples", key="load_samples"):
    paths, embs = download_and_embed_samples(embedder)
    if paths and embs is not None:
        current = set(st.session_state.image_paths)
        new_paths = []
        for p in paths:
            if p not in current:
                new_paths.append(p)
        if new_paths:
            idxs = []
            for i, p in enumerate(paths):
                if p in new_paths:
                    idxs.append(i)
            st.session_state.image_paths.extend(new_paths)
            to_add = embs[idxs]
            if st.session_state.doc_embeddings is None or st.session_state.doc_embeddings.size == 0:
                st.session_state.doc_embeddings = to_add
            else:
                st.session_state.doc_embeddings = np.vstack([st.session_state.doc_embeddings, to_add])
            st.success(f"Loaded {len(new_paths)} samples")
        else:
            st.info("Samples already loaded")
    else:
        st.error("Failed to load samples")

st.markdown("---")
st.subheader("Upload images or PDFs")

uploaded = st.file_uploader("Select files", type=["png","jpg","jpeg","pdf"],
                            accept_multiple_files=True, key="uploader", label_visibility="collapsed")
if uploaded:
    os.makedirs("uploaded_img", exist_ok=True)
    new_paths, new_embs = [], []
    for f in uploaded:
        if f.type == "application/pdf":
            p_paths, p_embs = process_pdf_file(f, embedder.image_embedding, base_output_folder="pdf_pages")
            if p_paths and p_embs:
                for path in p_paths:
                    if path in st.session_state.image_paths:
                        st.info(f"File '{os.path.basename(path)}' already uploaded")
                    else:
                        new_paths.append(path)
                new_embs.extend(p_embs)
        elif f.type in ["image/png","image/jpeg"]:
            path = os.path.join("uploaded_img", f.name)
            if path in st.session_state.image_paths:
                st.info(f"Image '{f.name}' already uploaded")
            else:
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                img = Image.open(path)
                emb = embedder.image_embedding(img)
                new_paths.append(path)
                new_embs.append(emb)
    if new_paths:
        st.session_state.image_paths.extend(new_paths)
        arr = np.vstack(new_embs) if new_embs else None
        if arr is not None:
            if st.session_state.doc_embeddings is None or st.session_state.doc_embeddings.size == 0:
                st.session_state.doc_embeddings = arr
            else:
                st.session_state.doc_embeddings = np.vstack([st.session_state.doc_embeddings, arr])
        st.success(f"Processed {len(new_paths)} files")

st.markdown("---")
st.subheader("Ask a question")

if not st.session_state.image_paths:
    st.warning("Load or upload images first")

question = st.text_input("Question", placeholder="e.g., What is Nike's net profit?")
run = st.button("Run", disabled=not (question and st.session_state.image_paths and st.session_state.doc_embeddings is not None))
st.markdown("### Results")

img_placeholder = st.empty()
ans_placeholder = st.empty()

if run:
    try:
        hits = find_top(question, embedder.text_embedding, st.session_state.doc_embeddings, st.session_state.image_paths, top_k=1)
        print("hits", hits)
        if hits:
            print("found", hits[0])
            path, score = hits[0]
            caption = f"Retrieved: {os.path.basename(path)} (score={score:.3f})"
            img_placeholder.image(path, caption=caption, use_container_width=True)
            if not openai_api_key:
                ans_placeholder.warning("Provide your OpenAI API key in the sidebar to generate an answer")
            else:
                ans = answer_with_gpt4o(question, path, openai_api_key)
                ans_placeholder.write(ans)
        else:
            img_placeholder.warning("No relevant image found")
            ans_placeholder.write("")
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption("CLIP for retrieval, GPT-4o for answering")