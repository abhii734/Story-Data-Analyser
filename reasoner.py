import os
import re
import random
import pickle

CACHE_FILE = "embeddings_cache.pkl"

class Reasoner:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.book_embeddings_cache = {}
        self.load_cache()

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.book_embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.book_embeddings_cache)} books from cache.")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def save_cache(self):
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.book_embeddings_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def retrieve_context(self, query, book_text, limit=15):
        """
        Hybrid context retrieval: 
        Combination of Semantic Search (Embeddings) and Entity/Keyword matching.
        """
        if not book_text:
            return []
            
        # 1. Smart Chunking
        raw_paragraphs = book_text.split('\n\n')
        chunks = []
        current_chunk = ""
        MIN_CHUNK_SIZE = 500
        MAX_CHUNK_SIZE = 3000
        
        for para in raw_paragraphs:
            if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:
                 current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # 2. Semantic Search (Try/Except for import)
        scored_chunks = []
        has_semantic = False
        
        if not hasattr(self, 'embedder'):
             try:
                 from sentence_transformers import SentenceTransformer
                 import numpy as np
                 from sklearn.metrics.pairwise import cosine_similarity
                 # Load model once
                 print("Loading embedding model (all-MiniLM-L6-v2)...")
                 self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                 self.has_embedder = True
             except ImportError:
                 print("sentence-transformers not installed. Falling back to keyword only.")
                 self.has_embedder = False

        if getattr(self, 'has_embedder', False):
            book_hash = hash(book_text[:1000]) # Hash first 1k chars to ID the book version
            
            if book_hash not in self.book_embeddings_cache:
                print(f"Embedding book chunks ({len(chunks)})...")
                embeddings = self.embedder.encode(chunks)
                self.book_embeddings_cache[book_hash] = (embeddings, chunks) # Store chunks too to map back!
                self.save_cache()
            else:
                embeddings, cached_chunks = self.book_embeddings_cache[book_hash]
                # Verify chunks length matches? If logic changed, this breaks.
                # Ideally hash the chunks too. But for hackathon, assume stable.
                # If cached_chunks != chunks, we need to re-encode.
                if len(cached_chunks) != len(chunks):
                     print("Cache mismatch (chunking changed). Re-embedding...")
                     embeddings = self.embedder.encode(chunks)
                     self.book_embeddings_cache[book_hash] = (embeddings, chunks)
                     self.save_cache()
            
            query_embedding = self.embedder.encode([query])
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            for idx, score in enumerate(similarities):
                scored_chunks.append([score, chunks[idx]]) 
            
            has_semantic = True

        # 3. Keyword/Entity Boosting
        keywords = [w.lower() for w in query.split() if w.isalnum() and len(w) > 3]
        entities = [w for w in query.split() if w[0].isupper() and len(w) > 3]
        
        if not scored_chunks: # If semantic failed, fill with 0s
             scored_chunks = [[0, c] for c in chunks]

        for i, item in enumerate(scored_chunks):
            # item is [score, chunk]
            chunk = item[1]
            chunk_lower = chunk.lower()
            
            keyword_score = 0
            for k in keywords:
                if k in chunk_lower:
                     keyword_score += 0.1 # Small boost for keywords
            
            for e in entities:
                if e in chunk:
                    keyword_score += 0.3 # Medium boost for entities
            
            # Add to semantic score
            item[0] += keyword_score
            
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return [c[1] for c in scored_chunks[:limit]]

    def evaluate_row(self, row, book_text=None, examples=[]):
        """
        Evaluates a single row from the test set.
        row: dict with keys 'book_name', 'char', 'content'
        book_text: full text of the book (optional)
        examples: list of dicts from train set for few-shot
        """
        book_name = row.get('book_name', 'Unknown Book')
        character = row.get('char', 'Unknown Character')
        content = row.get('content', '')
        
        # 1. Retrieve Context
        context_chunks = self.retrieve_context(content, book_text)
        context_str = "\n---\n".join(context_chunks) if context_chunks else "No specific book text content available. Use internal knowledge."

        # 2. Select Few-Shot Examples (Prioritize same book)
        same_book_examples = [ex for ex in examples if ex.get('book_name') == book_name]
        other_examples = [ex for ex in examples if ex.get('book_name') != book_name]
        
        # Take up to 3 examples
        selected_examples = same_book_examples[:2]
        if len(selected_examples) < 3:
            selected_examples.extend(other_examples[:3 - len(selected_examples)])
            
        examples_str = ""
        for ex in selected_examples:
            lbl = ex.get('label', 'unknown')
            val = 1 if lbl.lower() == 'consistent' else 0
            examples_str += f"Claim: {ex.get('content')}\nVerdict: {val}\nReasoning: (Example)\n\n"

        # 3. Construct Chain-of-Thought Prompt
        prompt = f"""
        You are a STRICT LITERARY JUDGE. Your job is to penalize inconsistencies.
        Target Book: {book_name}
        Character: {character}
        
        Task: Verify if the Claim is CONSISTENT (1) or CONTRADICTORY (0) with the provided Book Context.
        
        Claim: "{content}"
        
        Evidence (Retrieved Context):
        {context_str}
        
        JUDGING RULES:
        1. **Direct Contradiction**: If the book says X and the claim says Y (e.g., different dates, different killers, different locations for same event), the Verdict must be 0.
        2. **Timeline Conflict**: If the claim places an event in a time period where it is impossible based on the book, Verdict 0.
        3. **Characterization Conflict**: If the claim describes an action completely out of character (e.g., a pacifist committing murder without reason), Verdict 0.
        4. **Silence is Consistency**: If the book does NOT mention the events in the claim, and they don't contradict known facts, assume they happened "off-screen". Verdict 1.
        
        CRITICAL: Do not make excuses for the claim. If it gets a detail wrong, it is FALSE (0).
        
        Output format:
        REASONING: [Step-by-step comparison of Claim vs Evidence]
        VERDICT: [0 or 1]
        RATIONALE: [Concise final explanation]
        """
        
        response = self.llm.complete(prompt)
        
        # Parse response
        prediction = 1 # Default to consistent
        rationale = "Parser failed, assumed consistent."
        
        try:
            # Robust parsing for multi-line reasoning
            lines = response.split('\n')
            for line in lines:
                clean_line = line.strip()
                if clean_line.startswith("VERDICT:"):
                    val = clean_line.split("VERDICT:")[1].strip()
                    prediction = int(val) if val.isdigit() else 1
                if clean_line.startswith("RATIONALE:"):
                    rationale = clean_line.split("RATIONALE:")[1].strip()
            
            if rationale == "Parser failed, assumed consistent." and "REASONING:" in response:
                 rationale = "See reasoning."
                 
        except Exception as e:
            rationale = f"Error parsing response: {e}"
            
        return prediction, rationale

