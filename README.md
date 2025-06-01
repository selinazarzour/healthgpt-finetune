# 🩺 HealthGPT: A Fine-Tuned Medical Q&A Assistant

**HealthGPT** is a lightweight, domain-specific chatbot fine-tuned on medical question-answer pairs using the [TinyLLaMA-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model. It answers medically-relevant questions with more accuracy and specificity than generic LLMs thanks to supervised fine-tuning on curated healthcare data.

> 💡 Built and fine-tuned by Selina Zarzour as an MVP to explore domain adaptation of LLMs in healthcare and radiology.

---

## 🔍 Demo

🚀 Run it locally with a GPU
💬 Ask anything from “What does a CT scan show in pneumonia?” to “Can antibiotics treat viral infections?”

---

## 🧠 Fine-Tuning Details

- **Base model**: `TinyLLaMA-1.1B-Chat`
- **Dataset**: 16,000+ Q&A pairs from [MedQuad](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- **Method**: PEFT with LoRA + 4-bit quantization via `bitsandbytes`
- **Tools**: `transformers`, `datasets`, `trl`, `gradio`, `peft`, `bitsandbytes`

---

## 🏗️ Folder Structure

```
LLM-FineTuning-Project/
├── merged-model/          # Final full model with LoRA merged (used for inference)
├── lora-adapter/          # Optional: lightweight adapter weights
├── app.py                 # Gradio interface
├── requirements.txt       # Dependencies for deployment
├── README.md              # This file
└── training-notebook.ipynb  # Training and fine-tuning steps (optional)
```

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/selinaz/healthgpt-finetune
cd healthgpt-finetune

# (Optional) Create a virtual environment
pip install -r requirements.txt

python app.py
```

---

## ✍️ Example Prompts

- “What causes high blood pressure?”
- “Why is VFSS used instead of CT for swallowing studies?”
- “Can CT scans detect early signs of lung cancer?”
- “What does a hypodense liver lesion suggest?”
- “Is it safe to take vaccines during pregnancy?”

---

## 📦 Model Weights

- [Merged model on Hugging Face](https://huggingface.co/selinaz/HealthGPT-MedQA) (if pushed)
- [LoRA Adapter (optional)](https://huggingface.co/selinaz/HealthGPT-LoraAdapter)

---

## 🧪 Limitations

- Not a substitute for professional medical advice.
- Trained on general medical Q&A — may not reflect latest guidelines or rare conditions.
- Best used in research, education, or prototype exploration.

---

## 👩‍💻 Author

**Selina Zarzour**  
🧠 Interested in GenAI, healthcare innovation, and applied LLMs
