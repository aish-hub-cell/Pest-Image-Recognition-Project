🌿 Pest & Disease Recognition Using Deep Learning + GPT 🔍🧠

A bilingual AI-based tool that detects pests and plant diseases from images using **Convolutional Neural Networks (CNN)** and provides detailed solutions using **GPT-4** via `gpt4all`.

---

## 🧠 Description

This project leverages TensorFlow and deep learning to classify common plant diseases and pests from images. Once identified, a generative model (GPT) provides **treatment options, prevention tips, and extra knowledge** about the issue — making it a powerful tool for farmers, agri-students, and researchers.

> 💬 Results are provided in **structured expert-level responses**, with sections like causes, cures, home remedies, and prevention tips, both in English and Hindi.

---


## 💻 Features 

- 🧠 CNN-based image classification
- 🤖 GPT-powered solution generator
- 📸 Predict from image
- 🌐 Bilingual support (EN/HI)
- 🧪 Scientific and detailed plant health info

---

## 🛠️ Tech Stack

- TensorFlow / Keras
- GPT4All (`mistral-7b-instruct`)
- NumPy
- ImageDataGenerator
- HDF5, GZip

---

## 🧪 Classes Detected

- 🌿 Disease: Early Blight
- 🌿 Disease: Leaf Mold
- 🐛 Pest: Aphids
- 🌽 Pest: Corn Earworms
- 🕷️ Pest: Spider Mites

---

## 🚀 How to Run

### 1. Train the model (if needed)

```bash
python train_pest_model.py
```
### 2. Predict & Get Solution

```bash
python pest_recognition.py
```
### 3. Get Solution in Hindi

```bash
python pest_recognition_hindi.py 
```
## Dataset Structure

### Make sure your dataset follows this structure:

farm_insects/
├── Disease_Early Blight/
├── Disease_Leaf Mold/
├── Pest_Aphids/
├── Pest_Corn Earworms/
└── Pest_Spider Mites/

---

## 📜 License

This project is licensed under the MIT License.
