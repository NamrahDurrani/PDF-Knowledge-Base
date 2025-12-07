
import os
import json
import math
from typing import List, Tuple, Dict
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import hashlib
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def pdf_to_text(pdf_path: str) -> str:
    """Very simple extraction: read pages and join. For complex PDFs, use a better loader."""
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[Dict]:
    """
    Splits text into chunks of approx chunk_size tokens (words here), with overlap.
    Returns list of dict: {'id':..., 'text':..., 'meta': {...}}
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        meta = {"chunk_index": chunk_id}
        chunks.append({"id": _hash_text(chunk_text + str(chunk_id)), "text": chunk_text, "meta": meta})
        chunk_id += 1
        i += chunk_size - chunk_overlap
    return chunks


class KB:
    def __init__(
        self,
        index_path: str = "kb.index",
        meta_path: str = "kb_meta.json",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        hf_gen_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: int = -1,  # -1 => CPU, >=0 => GPU
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.embedding_model_name = embedding_model_name
        self.hf_gen_model = hf_gen_model
        self.device = device

        # load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)

    
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.index = None
        self.ids = []  
        self.metadict = {}  

        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.load()
                logger.info("Loaded existing KB from disk.")
            except Exception as e:
                logger.warning(f"Failed to load existing KB: {e}. Starting empty.")

        
        self.generator = None
        self.gen_tokenizer = None


    def save(self):
        """Persist faiss index and metadata."""
      
        logger.info("Saving FAISS index to %s", self.index_path)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "metadict": self.metadict}, f, ensure_ascii=False, indent=2)

    def load(self):
        """Load faiss index and metadata from disk."""
        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.ids = data["ids"]
            self.metadict = data["metadict"]

  
    def build_from_pdf(self, pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 200, persist: bool = True):
        """Full pipeline to build KB from a single PDF file."""
        logger.info("Extracting PDF text...")
        text = pdf_to_text(pdf_path)
        if not text.strip():
            raise ValueError("PDF contained no extractable text.")
        logger.info("Creating text chunks...")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info("Embedding %d chunks...", len(chunks))
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        logger.info("Creating FAISS index (InnerProduct on normalized vectors -> cosine). dim=%d", self.dim)
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        self.index = index
        self.ids = []
        self.metadict = {}
        for i, c in enumerate(chunks):
            cid = c["id"]
            self.ids.append(cid)
            self.metadict[cid] = {"text": c["text"], "meta": c["meta"], "source_pdf": os.path.basename(pdf_path)}
        if persist:
            self.save()
        logger.info("KB built with %d vectors.", len(self.ids))
        return len(self.ids)


    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Returns list of (text, score, meta) for top_k nearest chunks.
        Score is cosine similarity (because we normalized embeddings).
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Build or load a KB first.")
        q_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            cid = self.ids[int(idx)]
            md = self.metadict.get(cid, {})
            results.append((md.get("text", ""), float(score), md))
        return results


    def _ensure_generator(self):
        if self.generator is not None:
            return
        logger.info("Loading generation model and tokenizer: %s", self.hf_gen_model)
        try:
            self.gen_tokenizer = AutoTokenizer.from_pretrained(self.hf_gen_model, use_fast=True, trust_remote_code=True)
            self.gen_model = AutoModelForCausalLM.from_pretrained(
                self.hf_gen_model, device_map="auto" if self.device >= 0 else None, trust_remote_code=True
            )
            self.generator = pipeline(
                "text-generation",
                model=self.gen_model,
                tokenizer=self.gen_tokenizer,
                device=0 if self.device >= 0 else -1,
            )
        except Exception as e:
            logger.exception("Failed to load model with trust_remote_code=True; retrying without it. Error: %s", e)
            self.gen_tokenizer = AutoTokenizer.from_pretrained(self.hf_gen_model, use_fast=True)
            self.gen_model = AutoModelForCausalLM.from_pretrained(self.hf_gen_model)
            self.generator = pipeline(
                "text-generation",
                model=self.gen_model,
                tokenizer=self.gen_tokenizer,
                device=0 if self.device >= 0 else -1,
            )

    def _build_prompt(self, question: str, retrieved: List[Tuple[str, float, Dict]], max_context_chars: int = 2000) -> str:
        """
        Build a short but informative prompt:
        - system instructions
        - context: top retrieved chunks (concatenated, trimmed)
        - user question
        The generation model is instructed to be concise and give sources.
        """
        system = (
            "You are a concise helpful assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not contained in the context, say: 'I don't know — I couldn't find information in the document.' "
            "Be to-the-point, 2-6 short sentences max, and cite which chunk(s) you used (by index)."
        )

        context_parts = []
        for i, (text, score, md) in enumerate(retrieved):
            header = f"[chunk_{i} | score={score:.3f}]"
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 1000:
                snippet = snippet[:1000] + " ...[truncated]"
            context_parts.append(f"{header}\n{snippet}\n")
        context = "\n\n".join(context_parts)
        if len(context) > max_context_chars:
            context = context[: max_context_chars] + "\n...[context truncated]"

        prompt = (
            f"SYSTEM: {system}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"Answer:"
        )
        return prompt

    def answer_query(self, question: str, top_k: int = 5, generation_cfg: Dict = None) -> Tuple[str, List[Dict]]:
        """
        Main query function. Returns (answer_text, sources(list of dicts)).
        Each source dict: {'text':..., 'score':..., 'meta':...}
        """
        if self.index is None:
            raise RuntimeError("No KB loaded. Build KB first.")
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return "I don't know — the knowledge base is empty.", []

        prompt = self._build_prompt(question=question, retrieved=retrieved)

        self._ensure_generator()

        cfg = {"max_new_tokens": 256, "do_sample": False, "temperature": 0.0}
        if generation_cfg:
            cfg.update(generation_cfg)

        gen_out = self.generator(prompt, **cfg)
        raw_text = gen_out[0]["generated_text"]

        answer = raw_text
        marker = "Answer:"
        if marker in raw_text:
            answer = raw_text.split(marker, 1)[1].strip()
        if len(answer) > 1500:
            answer = answer[:1500] + " ...[truncated]"

        sources = []
        for i, (text, score, md) in enumerate(retrieved):
            sources.append({"index": i, "score": score, "text": text, "meta": md})
        return answer, sources
