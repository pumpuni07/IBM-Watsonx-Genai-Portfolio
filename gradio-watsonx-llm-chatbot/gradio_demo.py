"""Gradio demo: a sum calculator.

First Gradio app from the IBM Skills Network lab "Set Up a Simple Gradio
Interface to Interact with Your Models". gr.Number() creates numeric
input/output fields.
"""

import gradio as gr


def add_numbers(Num1, Num2):
    return Num1 + Num2


# Define the interface
demo = gr.Interface(
    fn=add_numbers,
    inputs=[gr.Number(), gr.Number()],  # two numeric input fields
    outputs=gr.Number(),                # numeric output field
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
