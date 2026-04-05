import typer

from agent.actions import ActionType
from agent.controller import AgentController
from llm.factory import create_llm_client
from tools.wikipedia_tool import WikipediaTool
from tools.unit_converter import UnitConverter
from tools.svt_solver import SVTSolver
from utils.logging import setup_logging
from config.settings import settings

import itertools
import threading
import time

def _spinner(stop_event: threading.Event) -> None:
    start = time.time()
    for char in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        if stop_event.is_set():
            break
        elapsed = time.time() - start
        print(f"\r{char} Thinking... ({elapsed:.1f}s)", end="", flush=True)
        time.sleep(0.1)
    print("\r" + " " * 30 + "\r", end="", flush=True)

app = typer.Typer(add_completion=False, help="CLI chat for the AI agent.")


def create_agent() -> AgentController:
    llm = create_llm_client()
    unit_converter = UnitConverter()
    tools = {
        ActionType.SEARCH_WIKIPEDIA: WikipediaTool(),
        ActionType.COMPUTE_SVT: SVTSolver(unit_converter),
    }
    return AgentController(llm=llm, tools=tools)


@app.callback(invoke_without_command=True)
def main():
    """Start interactive chat session."""
    setup_logging(level="DEBUG", verbose=settings.verbose)
    agent = create_agent()

    print("AI Agent started ({}). Type 'exit' to quit.\n".format(settings.model_name))

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if not settings.verbose:
            stop = threading.Event()
            t = threading.Thread(target=_spinner, args=(stop,), daemon=True)
            t.start()
            response = agent.run(user_input)
            stop.set()
            t.join()
        else:
            response = agent.run(user_input)

        print("Agent: {}\n".format(response))

if __name__ == "__main__":
    app()
