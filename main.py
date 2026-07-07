from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

class State(TypedDict):
    input: str
    output: str

def chatbot(state: State):
    response = llm.invoke(state["input"])
    return {"output": response.content}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

print("LangGraph Chatbot")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke({"input": user_input})
    print("AI:", result["output"])