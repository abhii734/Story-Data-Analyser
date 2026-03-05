import os
import pathway_ingest
from reasoner import Reasoner
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DATA_DIR = "./data"
BOOKS_DIR = os.path.join(DATA_DIR, "books")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
RESULTS_FILE = "results.csv"

class BackstoryConsistencyChecker:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.llm = None
        self.reasoner = None

    def initialize_llm(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if google_key:
            # gemini-2.5-flash is our best bet for speed/limits
            model_name = 'models/gemini-2.5-flash'
            try:
                print(f"Connecting to {model_name}...")
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                model = genai.GenerativeModel(model_name)
                # Test connection
                model.generate_content("Hi")
                print(f"Connected to {model_name}.")
                
                class GeminiAdapter:
                    def __init__(self, m_name):
                        self.model = genai.GenerativeModel(m_name)
                        
                    def complete(self, prompt):
                        import time
                        for attempt in range(3):
                            try:
                                response = self.model.generate_content(prompt)
                                if not response or not hasattr(response, 'text') or not response.text:
                                    # Fallback for safety/blocked
                                    return "VERDICT: 1\nRATIONALE: Safety filter block (assumed consistent)."
                                return response.text
                            except Exception as e:
                                if "429" in str(e) or "quota" in str(e):
                                    wait_time = 120 + (attempt * 60)
                                    print(f"Quota hit. Sleeping {wait_time}s...")
                                    time.sleep(wait_time)
                                    continue
                                print(f"LLM Error: {e}")
                                return "VERDICT: 1\nRATIONALE: API Error."
                
                self.llm = GeminiAdapter(model_name)
                return True
            except Exception as e:
                print(f"Failed to connect: {e}")
        
        print("Using MockLLM.")
        from mock_llm import MockLLM
        self.llm = MockLLM()
        return False

    def run(self):
        print("Starting Pipeline...")
        books_map = pathway_ingest.ingest_books(BOOKS_DIR)
        train_data = pathway_ingest.ingest_csv(TRAIN_FILE)
        test_data = pathway_ingest.ingest_csv(TEST_FILE)

        self.initialize_llm()
        self.reasoner = Reasoner(self.llm)
        
        # Load existing progress
        processed_ids = set()
        results = []
        if os.path.exists(RESULTS_FILE):
            try:
                with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        processed_ids.add(row['id'])
                        results.append([row['id'], row['prediction'], row['rationale']])
                print(f"Resuming from {len(processed_ids)} completed rows.")
            except:
                pass

        total = len(test_data)
        
        for i, row in enumerate(test_data):
            row_id = str(row.get('id'))
            if row_id in processed_ids:
                continue
                
            book_name = row.get('book_name')
            print(f"[{i+1}/{total}] ID {row_id} ({book_name})...")
            
            book_text = None
            if book_name in books_map:
                book_text = books_map[book_name]
            else:
                for b_key, b_val in books_map.items():
                    if book_name and book_name.lower() in b_key.lower():
                        book_text = b_val
                        break
            
            # 15s delay between requests to keep TPM low
            time.sleep(15)
            
            try:
                pred, rationale = self.reasoner.evaluate_row(row, book_text=book_text, examples=train_data)
                results.append([row_id, pred, rationale])
                
                # Save after every row
                with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'prediction', 'rationale'])
                    writer.writerows(results)
            except Exception as e:
                print(f"Failed ID {row_id}: {e}")

        print(f"Done! Results in {RESULTS_FILE}")

if __name__ == "__main__":
    checker = BackstoryConsistencyChecker(DATA_DIR)
    checker.run()
