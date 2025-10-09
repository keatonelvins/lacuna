# /// script
# dependencies = [
#     "gradio",
#     "openai",
# ]
# ///
import argparse

import gradio as gr
from openai import OpenAI


def chat_function(message, history, endpoint, model_name):
    """Chat function that communicates with local OpenAI-compatible endpoint."""
    client = OpenAI(api_key="EMPTY", base_url=f"http://{endpoint}/v1" if not endpoint.startswith("http") else endpoint)

    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(model=model_name, messages=messages, stream=True)

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                yield response

    except Exception as e:
        yield f"Error: {str(e)}\n\nPlease check your endpoint and model configuration."


def create_demo():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="LLM Chat") as demo:
        gr.Markdown("# LLM Chat Interface")

        with gr.Row():
            endpoint_input = gr.Textbox(label="Endpoint", value="0.0.0.0:8000", placeholder="localhost:8000")
            model_input = gr.Textbox(label="Model", value="Qwen/Qwen3-8B", placeholder="model-name")

        gr.ChatInterface(
            fn=chat_function,
            additional_inputs=[endpoint_input, model_input],
            chatbot=gr.Chatbot(height=600, show_copy_button=True),
            textbox=gr.Textbox(placeholder="Type your message...", container=False),
            stop_btn="Stop",
        )

    return demo


def main():
    """Main function to launch the Gradio app."""
    parser = argparse.ArgumentParser(description="Simple LLM chat UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
