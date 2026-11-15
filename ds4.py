from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Ты онлайн ассистент и должен отвечать на вопросы вежливо'
        ),
        (
            "human",
            '{text}'
        )
    ]
)

test_app = FastAPI()

# loaders = TextLoader('nnn.txt', encoding="utf-8")
# documents = loaders.load()
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10,
#     length_function=len
# )
#
# chunks = text_splitter.split_text(str(documents))
#
# doc = Document(
#     page_content="Motion web академия",
#     metadata={"source": "{documents}"}
# )

model = ChatOllama(
    model='llama3.2:latest',
    temperature=0
)

chain = prompt | model | StrOutputParser()


class QuestionSchema(BaseModel):
    question: str


@test_app.post('/questions/')
async def questions_for_answer(question: QuestionSchema):
    question = question.question.strip()
    return chain.invoke({'text': question})


if __name__ == '__main__':
    uvicorn.run(test_app, host='127.0.0.1', port=8000)