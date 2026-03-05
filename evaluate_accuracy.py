import os
import random
import pathway_ingest
from reasoner import Reasoner

# Configuration
DATA_DIR = "./data"
BOOKS_DIR = os.path.join(DATA_DIR, "books")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")

def evaluate_accuracy():
    print("Initializing Accuracy Check...")
    
    # Initialize LLM
    import google.generativeai as genai
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("GOOGLE_API_KEY missing.")
        return
        
    genai.configure(api_key=key)
    
    class GeminiAdapter:
        def __init__(self):
            self.model = genai.GenerativeModel('models/gemini-1.5-pro-preview-12-2025')
        def complete(self, prompt):
            try:
                return self.model.generate_content(prompt).text
            except Exception:
                return ""
    
    llm = GeminiAdapter()
    reasoner = Reasoner(llm)
    
    # Load Data
    books_map = pathway_ingest.ingest_books(BOOKS_DIR)
    train_data = pathway_ingest.ingest_csv(TRAIN_FILE)
    
    # Pick random subset
    sample_size = 20 # Keep it small for speed
    subset = train_data[:sample_size] # Just take first 20 for determinism or random.sample(train_data, sample_size)
    
    print(f"Evaluating accuracy on {len(subset)} samples...")
    
    correct = 0
    for i, row in enumerate(subset):
        book_name = row.get('book_name')
        book_text = books_map.get(book_name)
        if not book_text:
             for b_key, b_val in books_map.items():
                if book_name and book_name.lower() in b_key.lower():
                    book_text = b_val
                    break
        
        # Ground Truth
        label_str = row.get('label', '').lower()
        if 'consistent' in label_str:
            ground_truth = 1
        elif 'contradict' in label_str:
            ground_truth = 0
        else:
            continue # Skip unknown labels
            
        # Predict
        pred, _ = reasoner.evaluate_row(row, book_text=book_text, examples=[]) # Zero-shot or use train_data excluding current?
        # Ideally few-shot, but for speed/simplicity let's correct this.
        # Actually to measure TRUE performance we should use few-shot 
        # but exclude the current row.
        examples = [ex for ex in train_data if ex['id'] != row['id']]
        pred, rationale = reasoner.evaluate_row(row, book_text=book_text, examples=examples)
        
        if pred == ground_truth:
            correct += 1
        else:
            print(f"Mismatch ID {row.get('id')}: Pred {pred} vs Truth {ground_truth}")
            
    accuracy = (correct / len(subset)) * 100
    print(f"\nFinal Accuracy on {len(subset)} samples: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_accuracy()
