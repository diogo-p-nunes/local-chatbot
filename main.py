import sys
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import queue
import threading
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown

# Parse command line arguments
parser = argparse.ArgumentParser(description="Chatbot parameters")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--openai_api_key", type=str, default="EMPTY")
parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1")
parser.add_argument("--max_tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0)
args = parser.parse_args()

# Load model interface
model = ChatOpenAI(
    model=args.model,
    openai_api_key=args.openai_api_key,
    openai_api_base=args.openai_api_base,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    streaming=True,
)

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability. If you don't know the answer, just say 'I don't know'.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

# Define the agent graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
workflow.add_edge("model", END)

# Compile agent with memory for chat history
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

# Initialize console for rich output
console = Console()

def main() -> None:
    EXIT = "/bye"
    history: list = []

    while True:
        user_text = Prompt.ask("[bold magenta]>[/bold magenta]")
        if user_text.strip() == EXIT:
            console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")
            return

        history.append(HumanMessage(content=user_text))

        # --- run LangGraph ---
        console.print("[bold green]>[/bold green]", end=": ")
        full_response = ""
        for chunk, metadata in agent.stream(
            {"messages": history},
            {"configurable": {"thread_id": "abc123"}},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):
                full_response += chunk.content
                console.print(f"[green]{chunk.content}[/green]", end="")

        console.print("")
        history.append(AIMessage(content=full_response))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")