"""Lab exercise solution: combine two input sentences into one.

Two Textbox inputs, one Textbox output.
"""

import gradio as gr


def combine_sentences(sentence1: str, sentence2: str) -> str:
    """Join two sentences with a space, trimming stray whitespace."""
    return f"{sentence1.strip()} {sentence2.strip()}".strip()


demo = gr.Interface(
    fn=combine_sentences,
    inputs=[
        gr.Textbox(label="First sentence", lines=2, placeholder="Type the first sentence..."),
        gr.Textbox(label="Second sentence", lines=2, placeholder="Type the second sentence..."),
    ],
    outputs=gr.Textbox(label="Combined sentence"),
    title="Sentence Combiner",
    description="Combines two input sentences into one.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
