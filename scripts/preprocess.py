import re
import string

def clean_text(text):
    """
    Basic text cleaning: lowercasing, removing punctuation, and extra whitespaces.
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers (optional, but often helpful for sentiment/anxiety)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    sample = "I'm feeling EXTREMELY stressed! 123"
    print(f"Original: {sample}")
    print(f"Cleaned: {clean_text(sample)}")
