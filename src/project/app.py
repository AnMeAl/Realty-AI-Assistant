import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.project.search_engine import search_engine
from src.project.embeddings import get_query_multimodal_embedding, load_embedding_models
from src.project.report_generator import generate_report

st.set_page_config(page_title="Поиск квартир", layout="wide")

st.title("🏠 Умный поиск аренды")
st.markdown("Опишите квартиру, которую вы ищете, ИИ-ассистент найдет подходящие варианты.")

@st.cache_resource
def load_models():
    with st.spinner("Загрузка моделей..."):
        load_embedding_models()
        search_engine.load()
    return search_engine

search_engine = load_models()

query = st.text_area("Описание квартиры:", 
                      placeholder="Пример: Светлая двухкомнатная квартира рядом с метро, цена до 100 тыс. рублей",
                      height=100)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_button = st.button("Найти похожие квартиры", use_container_width=True)

if search_button and query:
    with st.spinner("Анализирую..."):
        query_embedding = get_query_multimodal_embedding(query)
        results = search_engine.search_hybrid(
                    query_embedding=query_embedding,
                    query_text=query,
                    top_k=5
                )
        
        if results:
            report = generate_report(query, results)
            st.markdown("---")
            st.markdown("## Анализ рынка")
            st.markdown(report)
        else:
            st.warning("Ничего не найдено. Попробуйте изменить описание.")