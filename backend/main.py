from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os

# Add scripts directory to path for preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from preprocess import clean_text

app = FastAPI(title="AI Exam Anxiety Detector API")

# Model path
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'anxiety_model.pt'))

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Global model variable
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run training script first.")
        
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

# Input schema
class PredictionRequest(BaseModel):
    text: str

# Labels map
label_map = {0: "Low Anxiety", 1: "Moderate Anxiety", 2: "High Anxiety"}

@app.post("/predict")
async def predict_anxiety(request: PredictionRequest):
    try:
        load_model()
        
        # Preprocess
        cleaned_text = clean_text(request.text)
        
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=64,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.softmax(outputs.logits, dim=1).flatten().tolist()
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        return {
            "anxiety_level": label_map[prediction],
            "probabilities": {
                "Low Anxiety": probs[0],
                "Moderate Anxiety": probs[1],
                "High Anxiety": probs[2]
            }
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "AI Based Exam Anxiety Detector API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
