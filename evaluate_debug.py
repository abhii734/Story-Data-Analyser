import os
import pathway_ingest
from reasoner import Reasoner
import google.generativeai as genai

# Configuration
DATA_DIR = "./data"
BOOKS_DIR = os.path.join(DATA_DIR, "books")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")

def evaluate_debug():
    print("Initializing Debug Check...")
    
    # Initialize LLM
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("GOOGLE_API_KEY missing.")
        return
        
    genai.configure(api_key=key)
    
    class GeminiAdapter:
        def __init__(self):
            # Using the working model
            self.model = genai.GenerativeModel('models/gemini-1.5-pro-preview-12-2025')
        def complete(self, prompt):
            try:
                return self.model.generate_content(prompt).text
            except Exception as e:
                return f"Error: {e}"
    
    llm = GeminiAdapter()
    
    # Create the Reasoner BUT we want to hook into it or just inspect
    reasoner = Reasoner(llm)
    
    # Load Data
    books_map = pathway_ingest.ingest_books(BOOKS_DIR)
    train_data = pathway_ingest.ingest_csv(TRAIN_FILE)
    
    # Filter for known contradictions (ground_truth = 0)
    # We want to see WHY we are predicting 1 for these.
    contradictions = [
        row for row in train_data 
        if 'contradict' in row.get('label', '').lower()
    ]
    
    # Take just 2-3 to analyze deeply
    subset = contradictions[:2]
    
    print(f"Deep analyzing {len(subset)} contradictions...")
    
    for i, row in enumerate(subset):
        row_id = row.get('id')
        print(f"\n=== DEBUGGING ID {row_id} ===")
        print(f"Claim: {row.get('content')}")
        
        book_name = row.get('book_name')
        book_text = books_map.get(book_name)
        if not book_text:
             for b_key, b_val in books_map.items():
                if book_name and book_name.lower() in b_key.lower():
                    book_text = b_val
                    break
        
        # 1. Inspect Retrieval
        # We manually call retrieve_context to see what the LLM gets
        context_chunks = reasoner.retrieve_context(row.get('content'), book_text)
        print(f"\n[Retrieved {len(context_chunks)} chunks]")
        for c_idx, chunk in enumerate(context_chunks[:2]): # Just print top 2
            print(f"-- Chunk {c_idx+1} (Start): {chunk[:100]}...")
            
        # 2. Inspect Prediction
        examples = [ex for ex in train_data if ex['id'] != row_id]
        pred, rationale = reasoner.evaluate_row(row, book_text=book_text, examples=examples)
        
        print(f"\nModel Verdict: {pred} (Expected: 0)")
        print(f"Model Rationale: {rationale}")
        
        if pred == 1:
            print("FAILURE: Model thinks this is consistent.")
        else:
            print("SUCCESS: Model caught the contradiction.")

if __name__ == "__main__":
    evaluate_debug()
