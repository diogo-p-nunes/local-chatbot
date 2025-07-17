import sys
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.utils import count_tokens_approximately
import queue
import threading
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text
import uuid

# Parse command line arguments
parser = argparse.ArgumentParser(description="Chatbot parameters")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--openai_api_key", type=str, default="EMPTY")
parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1")
parser.add_argument("--max_tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--stream", action="store_true")
args = parser.parse_args()

# Load model interface
model = ChatOpenAI(
    model=args.model,
    openai_api_key=args.openai_api_key,
    openai_api_base=args.openai_api_base,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    streaming=args.stream,
)

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Manage conversation history for no context overflow
memory = MemorySaver()
trimmer = trim_messages(
    max_tokens=800,
    strategy="last",
    token_counter=count_tokens_approximately,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define the function that calls the model
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages}
    )
    response = model.invoke(prompt)
    return {"messages": response}

# Define the agent graph; compile agent with memory for chat history
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
workflow.add_edge("model", END)
agent = workflow.compile(checkpointer=memory)

# Initialize console for rich output
console = Console()

def main() -> None:
    EXIT = "/bye"
    # each launch of the script gets a unique thread ID
    thread_id = str(uuid.uuid4()) 

    while True:
        user_text = Prompt.ask("[bold magenta]>[/bold magenta]")
        if user_text.strip() == EXIT:
            console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")
            return
        input_message = [HumanMessage(content=user_text)]

        # --- run LangGraph ---
        label = Text(">:", style="bold green")
        if args.stream:
            console.print(label, end=" ")
            for chunk, metadata in agent.stream(
                {"messages": input_message},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):
                    partial_answer = Text(chunk.content, style="green")
                    console.print(partial_answer, style="green", end="")
            console.print("")
        else:
            result = agent.invoke(
                {"messages": input_message},
                {"configurable": {"thread_id": thread_id}},
            )
            answer = Text(result["messages"][-1].content, style="green")
            console.print(label, answer)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold green]>[/bold green]: [green]Good-bye.[/green]")