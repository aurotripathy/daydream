import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from PIL import Image  
import os

auth_token = os.getenv("auth_token")
model_id = f"../elon_saved_model"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def infer(prompt, num_images=3):
    # prompt = [prompt] * num_images
    # images = pipe(prompt).images

    # grid = image_grid(images, rows=1, cols=3)    
    for i in range(num_images):
        yield pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images
    # return pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images


def infer_cumulative(prompt, num_images=5):
    images_list = []
    # grid = image_grid(images, rows=1, cols=3)    
    for _ in range(num_images):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        images_list.append(image)
        yield images_list
    # return pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images


with gr.Blocks() as demo:    
    with gr.Row() as row:
        with gr.Column():
            with gr.Row():
                input_text = gr.Textbox(lines=1, 
                                        label="Write a caption, don't worry, we'll imagine it for you.",
                                        value="A photo of a sks person")
                btn = gr.Button("Generate Pic")
            gallery = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
            ).style(grid=[5], height="auto")
    
        with gr.Column():
            hangman = gr.Textbox(
                label="Anchor Images",
            )
            used_letters_box = gr.Textbox(label="Used Letters")

    num_images = 5
    input_text.submit(infer_cumulative, inputs=[input_text], outputs=gallery)
    btn.click(infer_cumulative, [input_text], outputs=gallery)


demo.queue(concurrency_count=5, max_size=5).launch()

