# 🌐 English → Hindi Neural Machine Translation (Seq2Seq LSTM)

This project implements an **English to Hindi Neural Machine Translation (NMT) system** using a **Sequence-to-Sequence (Seq2Seq) LSTM architecture**, trained **with and without an Attention mechanism**, and deployed as an **interactive Streamlit web application**.

The project is designed to deeply understand:
- Seq2Seq learning
- Limitations of fixed context vectors
- How Attention improves long-sentence translation
- End-to-end ML workflow (training → inference → deployment)

---

## 🚀 Features

- ✅ Seq2Seq LSTM **without Attention**
- ✅ Seq2Seq LSTM **with Luong (dot-product) Attention**
- ✅ Teacher forcing during training
- ✅ Custom step-by-step inference pipeline
- ✅ Streamlit UI comparing both models
- ✅ Docker + Hugging Face deployment ready

---

## 🧠 Model Architectures

### 🔹 1. Seq2Seq without Attention

**Encoder**
- Embedding layer
- LSTM
- Outputs final hidden state (`h`) and cell state (`c`)

**Decoder**
- LSTM initialized using encoder’s final states
- Dense + Softmax over target vocabulary

📌 **Limitation**
- Entire input sentence is compressed into a **single fixed-size context vector**
- Translation quality degrades for longer sentences

---

### 🔹 2. Seq2Seq with Attention (Luong Attention)

This model addresses the fixed-context bottleneck by allowing the decoder to **attend to all encoder hidden states** at every decoding step.

**Attention computation (Luong-style):**
1. Alignment score  
   \[
   score_t = h_t^{decoder} \cdot h_s^{encoder}
   \]
2. Softmax → attention weights
3. Context vector = weighted sum of encoder hidden states
4. Concatenate context vector with decoder output
5. Final Dense + Softmax prediction

📌 **Advantages**
- Better handling of longer sentences
- Improved translation coherence
- Decoder dynamically focuses on relevant input words

---

## 📊 Dataset Details

- **Dataset:** IIT Bombay English–Hindi Parallel Corpus
- **Sampling:** Randomly sampled **100,000 sentence pairs**
- **Language Pair:** English → Hindi

### Sentence Lengths
| Language | Max Length | Padding |
|-------|-----------|---------|
| English | 30 | Post-padding |
| Hindi | 40 | Post-padding |

### Special Tokens
- `<sos>` – Start of sentence  
- `<eos>` – End of sentence  
- `<OOV>` – Out-of-vocabulary token  

---

## 🧹 Preprocessing Pipeline

### 🔸 English
- Lowercasing
- Tokenization using Keras `Tokenizer`
- Vocabulary size: **30,000**
- Padding: `post`
- Max length: **30**

### 🔸 Hindi
- Separate tokenizer
- Vocabulary size: **30,000**
- Decoder input shifted by one timestep (teacher forcing)
- Padding: `post`
- Max length: **40**

---

## 🏗️ Model Configuration

| Component | Value |
|--------|------|
| Embedding Dimension | **128** |
| LSTM Hidden Units | **256** |
| Encoder | LSTM (`return_sequences=True`, `return_state=True`) |
| Decoder | LSTM (`return_sequences=True`, `return_state=True`) |
| Attention Type | **Luong (Dot-Product)** |
| Vocabulary Size | 30,000 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 16 |
| Epochs | 10 |

---

## 🏋️ Training Details

- Teacher forcing used throughout training
- Entire dataset of **100,000 samples** used (no train–validation split)
- Decoder receives **ground-truth previous tokens**
- Attention weights computed using **encoder hidden states**

### 📈 Training Performance (With Attention)

| Epoch | Loss | Accuracy |
|-----|------|---------|
| 1 | ~1.86 | ~73% |
| 5 | ~0.90 | ~81% |
| 10 | ~0.50 | **~88%** |

---

## 🔍 Inference Pipeline

### ✳ Encoder Inference
- Input: padded English sentence
- Output:
  - Encoder hidden states (for attention)
  - Final hidden state
  - Final cell state

### ✳ Decoder Inference
- Generates one token at a time
- Uses:
  - Previous predicted token
  - Encoder hidden states
  - Previous decoder states
- Stops when `<eos>` is generated or max length is reached

📌 **Key Difference from Training**
- No teacher forcing
- Model predictions are fed back as inputs

---

## 🖥️ Streamlit Web Application

The Streamlit app allows:
- English sentence input
- Side-by-side translation comparison:
  - ❌ Without Attention
  - ✅ With Attention

### Tech Stack
- Streamlit
- TensorFlow / Keras
- NumPy
- Pickle

---

## 📦 Deployment

- Dockerized Streamlit app
- Hugging Face Spaces compatible
- CPU-only (no GPU required)

---

## 🧪 Sample Translations

| English Input | Without Attention | With Attention |
|--------------|------------------|---------------|
| I am happy | खुशी हो रही हूँ | खुशी हो रही हूँ |
| This is a book | यह एक किताब है | यह एक किताब है |
| I went to the market and bought fruits | ❌ degraded | ✅ more coherent |

---

## 📁 Project Structure

