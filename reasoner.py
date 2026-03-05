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

    def retrieve_context(self, query, book_text, limit=6):
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
            print(f"DEBUG: Book hash: {book_hash}")
            
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
        """
        book_name = row.get('book_name', 'Unknown Book')
        character = row.get('char', 'Unknown Character')
        content = row.get('content', '')
        
        # 1. Retrieve Context - Ensure we don't send too much
        context_chunks = self.retrieve_context(content, book_text)
        context_str = "\n---\n".join(context_chunks) if context_chunks else "No specific book text content found."

        # 2. Simplified prompt to reduce token count
        prompt = f"""
        Book: {book_name} | Character: {character}
        Claim: "{content}"
        
        Context Evidence:
        {context_str}
        
        Task: Is the claim CONSISTENT (1) or CONTRADICTORY (0) with the book?
        Consistency includes "off-screen" events that don't contradict the book.
        Contradictions include: wrong dates, wrong locations, or actions that violate book facts.
        
        Output:
        VERDICT: [0 or 1]
        RATIONALE: [Short explanation]
        """
        
        response = self.llm.complete(prompt)
        
        # Parse response
        prediction = 1 
        rationale = response
        
        try:
            for line in response.split('\n'):
                if "VERDICT:" in line:
                    val = line.split("VERDICT:")[1].strip()
                    prediction = int(val) if val.isdigit() else 1
                if "RATIONALE:" in line:
                    rationale = line.split("RATIONALE:")[1].strip()
        except:
            pass
            
        return prediction, rationale

