# /// script
# dependencies = ["gradio", "openai", "dotenv"]
# ///

import os
import sys
from pathlib import Path
from functools import partial

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOC = """
Chat with models after running `transformers serve` (for any model) or `uv run serve <model_path>`.

`transformers serve` will auto-list any model in the weights/ folder, but also accepts custom names/paths.

Usage:
  uv run chat
  uv run chat --api-base http://remote:8000/v1
"""


def get_local_models() -> list[Path]:
    """Get list of models from weights/ folder."""
    weights_dir = Path("weights")
    if not weights_dir.exists():
        return []

    models = []
    for folder in weights_dir.iterdir():
        if folder.is_dir():
            if (folder / "config.json").exists():
                models.append(weights_dir / folder.name)
            else:
                for subfolder in folder.iterdir():
                    if subfolder.is_dir() and (subfolder / "config.json").exists():
                        models.append(weights_dir / folder.name / subfolder.name)
    return sorted(models)


def chat(message: str, history: list[dict], model: str, client: OpenAI):
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    messages.append({"role": "user", "content": message})
    response = ""
    for chunk in client.chat.completions.create(model=model, messages=messages, stream=True):
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(DOC.strip())
        exit(0)

    api_base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
    client = OpenAI(base_url=api_base)

    try:
        models = client.models.list().data
    except Exception:
        print(
            f"Could not connect to {api_base}. Start server with `uv run transformers serve` or `uv run serve <model>`"
        )
        exit(1)

    if len(models) == 1:
        with gr.Blocks() as demo:
            gr.Markdown(f"**Model:** {models[0].id}")
            gr.ChatInterface(
                fn=partial(chat, model=models[0].id, client=client),
                type="messages",
                chatbot=gr.Chatbot(type="messages", allow_tags=["think"]),
            )
        demo.launch()
    else:
        with gr.Blocks() as demo:
            local_models = get_local_models()
            model_input = gr.Dropdown(
                label="Model",
                choices=local_models,
                allow_custom_value=True,
                value=local_models[0] if local_models else None,
            )
            gr.ChatInterface(
                fn=partial(chat, client=client),
                type="messages",
                additional_inputs=[model_input],
                chatbot=gr.Chatbot(type="messages", allow_tags=["think"]),
            )
        demo.launch()
