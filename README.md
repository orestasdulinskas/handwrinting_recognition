# ✍️ From Pen to Pixel: Custom OCR Pipeline for Handwritten Journal Digitization

This project is a handcrafted end-to-end **Optical Character Recognition (OCR)** pipeline built to transcribe my handwritten journal entries into digital text—using PyTorch, Faster R-CNN, AWS Lambda, and iOS Shortcuts. It's a personal and technical showcase of deep learning, MLOps, and full-stack AI deployment.

---

## 🚀 What This Project Does

🔹 Uses a **CNN-LSTM OCR model** trained from scratch on 60+ pages of my handwritten journals  
🔹 Fine-tunes **FasterRCNN_ResNet50** to automate bounding box annotation  
🔹 Chains both into an **inference pipeline** that extracts, segments, and transcribes handwriting  
🔹 Wraps the pipeline in an **API deployed via AWS Lambda & API Gateway**  
🔹 Accesses the API via a native **iOS Shortcut app** for mobile transcription

---

## 📄 Full Technical Report

Want to dive deep into the models, training process, architecture, and deployment stack?

👉 [Read the full PDF report](./From%20Pen%20to%20Pixel_%20Building%20a%20Custom%20OCR%20Pipeline%20to%20Digitize%20My%20Handwritten%20Journal%20Using%20PyTorch%20and%20AWS%20-%20Orestas%20Dulinskas.pdf)

Covers:
- Motivation
- Data Generation and Preparation
- OCR model architecture (CNN + BiLSTM + CTC)
- Annotation model (Faster R-CNN with transfer learning)
- AWS Lambda containerized deployment
- Inference pipeline logic
- iOS Shortcut integration and demo
- Results, CER/WER, error correction
- Challenges, learnings, and future work

---

## 📂 Key Files in This Repo

| File/Notebook | Description |
|---------------|-------------|
| `ocr_model.ipynb` | Trains the CNN-LSTM OCR model from scratch |
| `auto_annotator_model.ipynb` | Fine-tunes Faster R-CNN for line detection |
| `inference.ipynb` | Full pipeline: detection + OCR + decoding |
| `lambda_function.py` | AWS Lambda handler with integrated pipeline |
| `ios_app_pipeline.png` | Visual of iOS Shortcut interacting with the API |
| `From Pen to Pixel ... .pdf` | 📄 Full project report with all technical details |

---

## ⚙️ Tech Stack

- 🧠 PyTorch, Albumentations, TextBlob
- 📦 AWS Lambda (Dockerized), API Gateway, S3, ECR
- 📱 iOS Shortcuts for mobile interface
- 📷 VGG Image Annotator (for labeling training data)

---

## 🧪 Results

| Metric | Value |
|--------|-------|
| Character Error Rate (CER) | 2.3% |
| Word Error Rate (WER)      | 9.33% |
| Average Inference Time     | ~18s (CPU, Lambda) |
| Manual Transcription Time  | ~5 min/page ⏱️ |

✅ ~17x improvement in processing time  
✅ Fully automated pipeline  
✅ Self-trained on personal dataset (60+ A5 pages)

---

## 💡 Future Improvements

- Replace Faster R-CNN with YOLOv8 or DETR
- Move inference to GPU-backed container for speed
- Integrate LLM-based grammar + spell checking
- Create auto blog upload pipeline from transcribed text

---

## 👤 About Me

**Orestas Dulinskas**  
MSc Data Science | AI + MLOps Engineer-in-Progress  
[LinkedIn](https://www.linkedin.com/in/orestas-dulinskas) | [orestasdulinskas@gmail.com](mailto:orestasdulinskas@gmail.com)

---

> If you're building real-world AI products—or want to—I'm always open to connect.
