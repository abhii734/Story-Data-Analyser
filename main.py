import os
import pathway_ingest
from reasoner import Reasoner
import csv

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
            models_to_try = [
                'models/gemini-1.5-flash',
                'models/gemini-2.0-flash',
                'models/gemini-1.5-pro'
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"Attempting to connect to Google {model_name}...")
                    import google.generativeai as genai
                    genai.configure(api_key=google_key)
                    model = genai.GenerativeModel(model_name)
                    # Use a very short test to avoid burning quota
                    model.generate_content("Hi")
                    print(f"Google {model_name} connected successfully.")
                    
                    class GeminiAdapter:
                        def __init__(self, m_name):
                            self.model = genai.GenerativeModel(m_name)
                            
                        def complete(self, prompt):
                            import time
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    response = self.model.generate_content(prompt)
                                    if not response or not response.text:
                                        raise Exception("Empty response or Safety Blocked")
                                    return response.text
                                except Exception as e:
                                    err_str = str(e).lower()
                                    if ("429" in err_str or "quota" in err_str or "blocked" in err_str) and attempt < max_retries - 1:
                                        wait_time = 80 + (attempt * 40)
                                        print(f"Rate limited or blocked. Waiting {wait_time}s...")
                                        time.sleep(wait_time)
                                        continue
                                    print(f"[Gemini Error]: {e}")
                                    return "VERDICT: 0\nRATIONALE: Error or safety block from LLM."

                    self.llm = GeminiAdapter(model_name)
                    return True
                except Exception as e:
                    print(f"Warning: {model_name} failed ({e}).")
        
        print("Falling back to MockLLM.")
        from mock_llm import MockLLM
        self.llm = MockLLM()
        return False

    def run(self):
        print("Starting Consistency Checker Pipeline...")
        
        books_map = pathway_ingest.ingest_books(BOOKS_DIR)
        print(f"Loaded {len(books_map)} books.")
        
        try:
            train_data = pathway_ingest.ingest_csv(TRAIN_FILE)
            test_data = pathway_ingest.ingest_csv(TEST_FILE)
        except Exception as e:
            print(f"Error loading CSVs: {e}")
            return

        self.initialize_llm()
        self.reasoner = Reasoner(self.llm)
        
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Processing test cases (Sequential)...")
        results = []
        
        def process_row(args):
            i, row, total_cnt = args
            row_id = row.get('id')
            book_name = row.get('book_name')
            
            book_text = None
            if book_name in books_map:
                book_text = books_map[book_name]
            else:
                for b_key, b_val in books_map.items():
                    if book_name and book_name.lower() in b_key.lower():
                        book_text = b_val
                        break
            
            print(f"[{i+1}/{total_cnt}] Evaluating ID {row_id} (Book: {book_name})...")
            
            try:
                time.sleep(15) # Throttle to stay safe
                prediction, rationale = self.reasoner.evaluate_row(
                    row, 
                    book_text=book_text, 
                    examples=train_data
                )
                return [row_id, prediction, rationale]
            except Exception as e:
                print(f"Error processing ID {row_id}: {e}")
                return [row_id, 1, f"Error: {e}"]

        total = len(test_data)
        tasks = [(i, row, total) for i, row in enumerate(test_data)]
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_row, t) for t in tasks]
            
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                print(f"DEBUG: Finished task {res[0]}, total: {len(results)}/60")
                # Immediate save for crash recovery
                try:
                    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['id', 'prediction', 'rationale'])
                        writer.writerows(results)
                except:
                    pass
            
        print(f"Done! Final results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    checker = BackstoryConsistencyChecker(DATA_DIR)
    checker.run()
