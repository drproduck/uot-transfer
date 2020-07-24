import gradio as gr
from color_transfer import color_transfer

def transfer(img1, img2):
    img1_transformed, _ = color_transfer(img1, img2)
    return img1_transformed

gr.Interface(
    transfer, 
    [gr.inputs.Image(image_mode="RGB", label='source'), gr.inputs.Image(image_mode="RGB", label='target')],
    [gr.outputs.Image(label='transferred source')]
).launch(share=True)
