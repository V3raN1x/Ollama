from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


data_path = Path(__file__).with_name("nnn.txt")
documents = TextLoader(str(data_path), encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)
context = "\n\n".join(chunk.page_content for chunk in chunks)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — Pokémon AI Assistant, помощник по вселенной Pokémon. "
            "Отвечай кратко и понятно, используя только информацию из контекста. "
            "Рассказывай о покемонах, их типах, эволюциях, способностях, "
            "характеристиках, тренерах и регионах. "
            "Не придумывай характеристики и факты. "
            "Если вопрос задан по-русски, отвечай по-русски; "
            "если на кыргызском — отвечай на кыргызском. "
            "Если ответа нет в контексте, честно скажи, "
            "что такой информации пока нет."
        ),
        ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
    ]
)

model = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

chain = prompt | model | StrOutputParser()

app = FastAPI(title="Pokémon AI Assistant")


class QuestionSchema(BaseModel):
    question: str


@app.post("/questions/")
async def questions_for_answer(question: QuestionSchema):
    answer = chain.invoke(
        {
            "context": context,
            "question": question.question.strip(),
        }
    )

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001    )
