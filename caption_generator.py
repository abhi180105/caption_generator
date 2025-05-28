import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption
def generate_caption(image):
    if image is None:
        return "Please upload an image!"

    # Convert image to RGB
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

# Launch Gradio UI
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Caption Generator",
    description="Upload an image and get an AI-generated caption. Powered by BLIP by Salesforce 🤖"
)

interface.launch()

