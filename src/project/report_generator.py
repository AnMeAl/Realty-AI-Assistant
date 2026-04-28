from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.project.config import OPENAI_API_KEY, OPENAI_API_BASE
from src.project.search_engine import SearchResult
from typing import List


def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE
    )


def generate_report(query_text: str, results: List[SearchResult]) -> str:
    if not results:
        return "Похожих квартир не найдено. Попробуйте изменить описание."
    
    llm = get_llm()
    
    alternatives = []
    for i, r in enumerate(results, 1):
        alt = f"""| {i} | {r.price} | {r.area:.0f} | {r.rooms} | {r.floor}/{r.total_floors} | {r.address[:50]} | |"""
        alternatives.append(alt)
    
    context = "\n".join(alternatives)
    
    prompt = ChatPromptTemplate.from_template("""Ты — аналитик недвижимости. Твоя задача — сравнить найденные варианты объективно.

## Запрос пользователя:
{query}

## Найденные похожие квартиры:
{context}

## Выдай ответ в следующем формате:

*Общая картина:*
[2 предложения о том, какие цены на рынке и что можно ожидать]

**Сравнение вариантов:**
| № | Цена (руб.) | Площадь (м²) | Комнат | Этаж | Адрес | Оценка |
|---|--------------|--------------|--------|------|-------|--------|
| 1 | X | X | X | X | X | [*хорошая* / *переплата* / *спорная*, почему] |
| 2 | X | X | X | X | X | ... |
| 3 | X | X | X | X | X | ... |
| 4 | X | X | X | X | X | ... |
| 5 | X | X | X | X | X | ... |

**Сводка:**
- *Самый дешёвый*: вариант №...
- *Самый дорогой*: вариант №...
- *Лучшее соотношение цена/качество*: вариант №...
- *Стоит ли рассматривать аналоги*: [да/нет, почему]

**Итоговый вывод:**
[1 предложение с чёткой рекомендацией]

Отвечай строго по формату. Будь объективным, не приукрашивай. Основывайся только на загруженных данных!
""")
    
    response = llm.invoke(prompt.format_messages(query=query_text, context=context))
    return response.content