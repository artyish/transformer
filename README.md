# 🧠 Transformer From Scratch (Low-Level, Decoder-Only)

Welcome! This project is a low-level implementation of a **decoder-only Transformer model** built entirely from scratch using **TensorFlow** and **raw Python** no Hugging Face, no Keras shortcuts.

> 🔍 Built to deeply understand how models like GPT work under the hood — from embeddings and attention to training and inference.

---

## 🚀 Project Highlights

- ✅ **Manual Multi-Head Self-Attention**
- ✅ **Sinusoidal Positional Encoding**
- ✅ **Custom Tokenizer + Vocabulary** from 4,000+ SQuAD QA pairs
- ✅ **Feedforward + LayerNorm** Implementation
- ✅ **End-to-End Training Pipeline** (with checkpointing, batching)
- ✅ **Inference Loop for Next Token Prediction**
- ✅ **~661,000 Trainable Parameters**

---

## ⚠️ Heads Up!

> 🧪 **This model uses only a single Transformer layer.**  
> It’s built for learning, not production use — so don’t expect GPT-level responses.  
> The goal here was **understanding**, not state-of-the-art performance.

---

## 🛠️ Model Architecture

- **Embedding Layer** → map tokens to 32-dimensional vectors
- **Positional Encoding** → add temporal context
- **Multi-Head Self-Attention** → learn inter-token relationships
- **Feedforward Network** → non-linearity + projection
- **Final Linear Layer** → vocab-sized logits for token prediction

---

## 🧪 Dataset

- Uses the [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) dataset
- Only the **first 4000 QA pairs** used for building vocabulary and input training sequences
- Each input is tokenized and converted to integer sequences for training

---

## 📈 Training Details

- **Optimizer**: Adam  
- **Loss**: SparseCategoricalCrossentropy (from logits)  
- **Batch Size**: 128  
- **Sequence Length**: 20  
- **Training Epochs**: 100  
- **Device**: RTX 3050 (6GB Laptop GPU)

---

## 🔎 Transformer Architecture

You can run the inference loop using:
python inference.py
## 📸 Screenshot
![Transformer Architecture](https://github.com/artyish/transformer/blob/main/screenshots/diagram.png)

## 📂 Folder Structure
```
transformer/
├── attentionhead.py            # Multi-head attention logic
├── forward_network.py          # Feedforward + LayerNorm block
├── train_model.py              # Training loop
├── inference.py                # Inference loop
├── utilities/
│   ├── tokenizer.py            # Tokenizer + vocab builder
│   └── positional_encodings.py # Sinusoidal positional encoding
├── checkpoints/                # TensorFlow checkpoints
```
## 📜 License
This project is open-source under the MIT License.





