"""
Need atleast a RTX 3090 class 24 GB machine
"""
import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from PIL import Image  
import os
from PIL import Image

auth_token = os.getenv("auth_token")
model_id = f"../elon_saved_model"
static_images_dir = '../images/elon'
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
num_images = 3

def render():
    file_list = os.listdir(static_images_dir)
    print(file_list)
    # anchors = []
    # with open (static_images_dir, 'r') as f:
    #     anchors.append()


def infer_cumulative(prompt, num_images=num_images):
    images_list = []

    for _ in range(num_images):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        images_list.append(image)
        yield images_list


with gr.Blocks() as demo:    
    with gr.Row() as row:
        with gr.Column(): ## Left
            with gr.Row():
                input_text = gr.Textbox(lines=1, 
                                        label="Complete the caption, we'll daydream it for you.",
                                        value="A photo of a sks person")
                btn = gr.Button("Day Dream")
            gallery = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
            ).style(grid=1, height=512, container=True)

    
        with gr.Column():  # Right
            static_gallery = gr.Gallery(render,
                label="Anchor images", show_label=True, elem_id="gallery"
            ).style(grid=1, height=256, container=True)
            

    # Render the static piece as thumbnails

    input_text.submit(infer_cumulative, inputs=[input_text], outputs=gallery)
    btn.click(infer_cumulative, [input_text], outputs=gallery)


demo.queue(concurrency_count=num_images, max_size=num_images).launch()

