from hloc_glomap.gradio_ui import process_block
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab(label="Process"):
        process_block.render()

demo.launch()
