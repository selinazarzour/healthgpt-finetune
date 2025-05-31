import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr

# Load model and tokenizer
model_path = "/merged-model"
original_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(original_model_id)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config)

# Define generation function
def generate_answer(question):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# Launch UI
gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question...", label="Your Question"),
    outputs=gr.Textbox(label="HealthGPT Answer"),
    title="🩺 HealthGPT: Fine-Tuned Medical Assistant",
    description="Ask any medical question and get a domain-specific response from a fine-tuned LLM.",
).launch()