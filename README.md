# AI Based Exam Anxiety Detector

This project is an AI system that detects exam anxiety levels from student-written text using Natural Language Processing and a fine-tuned BERT model.

## Features
- **Dataset**: 80+ sample student sentences labeled as Low, Moderate, or High Anxiety.
- **BERT Model**: Fine-tuned `bert-base-uncased` for text classification.
- **FastAPI Backend**: Provides a `/predict` endpoint for real-time inference.
- **Streamlit Frontend**: A user-friendly web interface for anxiety analysis with visual indicators and calming tips.

## Project Structure
```text
AI_Exam_Anxiety_Detector/
├── dataset/
│   └── anxiety_dataset.csv
├── model/
│   └── (anxiety_model.pt will be generated here)
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── scripts/
│   ├── train_model.py
│   └── preprocess.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to fine-tune the BERT model on the provided dataset. This will save the model to `model/anxiety_model.pt`.
```bash
cd scripts
python train_model.py
```
*Note: This might take a few minutes depending on your CPU/GPU.*

### 3. Run FastAPI Backend
Start the backend server to handle inference requests:
```bash
python backend/main.py
```
The API will be available at `http://localhost:8000`.

### 4. Run Streamlit Frontend
Open a new terminal and start the web interface:
```bash
streamlit run frontend/app.py
```

### 5. Test the System
- Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).
- Enter a sentence like "I am feeling extremely stressed about my exams tomorrow".
- Click "Analyze Anxiety" to see the result.

## Ethical Note
This system is designed strictly as a supportive and non-diagnostic tool. It aims to help students become aware of their emotional state and provide general calming tips.
