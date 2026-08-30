"""Streamlit application for conversational product retrieval."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from shopping_search.assistant import ShoppingAssistant
from shopping_search.catalog import CatalogIndex
from shopping_search.retrieval import QueryEncoder


st.set_page_config(page_title="VLM Shopping Search", layout="wide")
st.title("VLM과 속성 정보를 활용한 쇼핑 검색")


@st.cache_resource
def load_resources(index_dir: str):
    catalog = CatalogIndex(Path(index_dir))
    encoder = QueryEncoder(catalog.metadata["model_name"], catalog.metadata["pretrained"])
    assistant = ShoppingAssistant()
    return catalog, encoder, assistant


image_root = Path(os.environ.get("SHOPPING_IMAGE_ROOT", "data/catalog"))
index_dir = os.environ.get("SHOPPING_INDEX_DIR", "artifacts/index")

try:
    catalog, encoder, assistant = load_resources(index_dir)
except Exception as exc:
    st.error(f"앱 설정 또는 인덱스를 확인하세요: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dialogue" not in st.session_state:
    st.session_state.dialogue = ""
if "results" not in st.session_state:
    st.session_state.results = []
if "timings" not in st.session_state:
    st.session_state.timings = {}

left, right = st.columns([1, 2])
with left:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    user_input = st.chat_input("찾고 싶은 상품을 설명하세요")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.dialogue += f"\nUser: {user_input}"
        started = time.perf_counter()

        summary_started = time.perf_counter()
        summary = assistant.summarize(st.session_state.dialogue)
        summary_time = time.perf_counter() - summary_started

        embedding_started = time.perf_counter()
        query = encoder.encode(summary)
        embedding_time = time.perf_counter() - embedding_started

        search_started = time.perf_counter()
        results, cluster_id = catalog.search(query, candidate_k=100, display_k=10)
        search_time = time.perf_counter() - search_started

        question_started = time.perf_counter()
        question = assistant.generate_valid_question(
            summary, catalog.cluster_description(cluster_id), st.session_state.dialogue
        )
        question_time = time.perf_counter() - question_started

        assistant_text = f"검색 요약: {summary}"
        if question:
            assistant_text += f"\n\n{question}"
            st.session_state.dialogue += f"\nAssistant: {question}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.results = results
        st.session_state.timings = {
            "요약": summary_time,
            "임베딩": embedding_time,
            "검색": search_time,
            "질문": question_time,
            "전체": time.perf_counter() - started,
        }
        st.rerun()

with right:
    st.subheader("추천 상품 Top 10")
    if st.session_state.results:
        columns = st.columns(5)
        for position, result in enumerate(st.session_state.results):
            path = image_root / result.relative_path
            with columns[position % 5]:
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning(f"이미지 없음: {result.relative_path}")
                st.caption(f"{Path(result.relative_path).name} · {result.score:.3f}")
    if st.session_state.timings:
        st.subheader("최근 처리 시간")
        st.json({name: f"{seconds:.2f}s" for name, seconds in st.session_state.timings.items()})

