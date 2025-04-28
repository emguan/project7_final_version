import gradio as gr
from transformers import pipeline

# Load the fill-mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

def predict(sentence):
    if '[MASK]' not in sentence:
        return "Error: You must include [MASK] in your sentence."
    
    results = fill_mask(sentence)
    output = ""
    for i, prediction in enumerate(results):
        token = prediction['token_str']
        score = prediction['score']
        output += f"{i+1}. **{token}** — {score:.4f}\n"
    return output

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence with [MASK]..."),
    outputs="markdown",
    title="BERT Masked Word Predictor",
    description="Enter a sentence containing [MASK]. Example: 'The capital of France is [MASK].' BERT will predict the most likely words!"
)

iface.launch()
