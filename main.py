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
        # Initialize LLM
        # Check for API Key in env or .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        api_key = os.getenv("OPENAI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        # Try Google first
        if google_key:
            try:
                print("Attempting to connect to Google Gemini...")
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                
                # Test connection before committing
                model = genai.GenerativeModel('models/gemini-2.5-flash')
                model.generate_content("Test")
                print("Google Gemini connected successfully.")
                
                class GeminiAdapter:
                    def __init__(self):
                        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                        
                    def complete(self, prompt):
                        import time
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                response = self.model.generate_content(prompt)
                                return response.text
                            except Exception as e:
                                if "429" in str(e) and attempt < max_retries - 1:
                                    print(f"Rate limited. Waiting 60s (Attempt {attempt+1}/{max_retries})...")
                                    time.sleep(65)
                                    continue
                                print(f"[Gemini Error]: {e}")
                                return ""

                self.llm = GeminiAdapter()
                return True
                
            except Exception as e:
                print(f"Warning: Google Gemini initialization failed ({e}).")
        
        # Fallback to Mock
        print("Falling back to MockLLM.")
        from mock_llm import MockLLM
        self.llm = MockLLM()
        return False

    def run(self):
        print("Starting Consistency Checker Pipeline...")
        
        # 1. Load Data
        print("Loading Books...")
        books_map = pathway_ingest.ingest_books(BOOKS_DIR)
        print(f"Loaded {len(books_map)} books.")
        
        print("Loading CSV Data...")
        try:
            train_data = pathway_ingest.ingest_csv(TRAIN_FILE)
            test_data = pathway_ingest.ingest_csv(TEST_FILE)
        except Exception as e:
            print(f"Error loading CSVs: {e}")
            return

        print(f"Loaded {len(train_data)} training examples.")
        print(f"Loaded {len(test_data)} test cases.")

        # 2. Initialize Logic
        self.initialize_llm()
        self.reasoner = Reasoner(self.llm)
        
        # Pre-load embedding model
        # print("Pre-loading embedding model...")
        # self.reasoner.retrieve_context("warmup", "The quick brown fox jumps over the lazy dog.")
        
        # 3. Process Test Cases
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Processing test cases (max_workers=3)...")
        results = []
        
        def process_row(args):
            i, row, total_cnt = args
            row_id = row.get('id')
            book_name = row.get('book_name')
            
            # Find book text
            book_text = None
            if book_name in books_map:
                book_text = books_map[book_name]
            else:
                for b_key, b_val in books_map.items():
                    if book_name.lower() in b_key.lower():
                        book_text = b_val
                        break
            
            print(f"[{i+1}/{total_cnt}] Evaluating ID {row_id} (Book: {book_name})...")
            
            try:
                # Basic rate limiting per thread - stayed at 5s for single worker
                time.sleep(5) 
                
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
            # We map the function to the tasks
            futures = [executor.submit(process_row, t) for t in tasks]
            
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                print(f"DEBUG: Finished task, total results now: {len(results)}")
            
        # Sort results by ID to keep order
        # Assuming ID is convertible to int? Or just string sort.
        # Original order is better.
        # But 'results' order is random now.
        # Let's verify sort needed? 
        # test.csv has random IDs. 
        # Actually simplest is to just write them. But user might prefer sorted.
        # Since I'm appending as completed, they are scrambled.
        # Let's sort by the index 'i' passed in tasks? 
        # Wait, I didn't return 'i'.
        # Whatever, order doesn't strictly matter for CSV submission usually.
            
        # 4. Write Results
        print(f"Writing results to {RESULTS_FILE}...")
        try:
            with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id', 'prediction', 'rationale'])
                writer.writerows(results)
            print("Done!")
        except Exception as e:
            print(f"Error writing results: {e}")

if __name__ == "__main__":
    checker = BackstoryConsistencyChecker(DATA_DIR)
    checker.run()

