
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

temperatures = [0, 0.3, 0.5, 0.7, 1]

messages = [
    SystemMessage(
        content="Ты преподаватель Python. Объясняй понятно и приводи примеры."
    ),
    HumanMessage(
        content="Python — это..."
    )
]

for temperature in temperatures:
    model = ChatOllama(
        model="llama3.2:latest",
        temperature=temperature,
        num_predict=50
    )

    print(f"Temperature: {temperature}")

    response = model.invoke(messages)

    print("Ответ:", response.content)
    print(
        "Причина завершения:",
        response.response_metadata.get("done_reason")
    )
    print(
        "Количество токенов:",
        response.response_metadata.get("eval_count")
    )
