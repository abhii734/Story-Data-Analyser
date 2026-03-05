import random

class MockLLM:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt, **kwargs):
        """
        Simulate an LLM completion.
        """
        # print(f"[MockLLM] Processing prompt (len={len(prompt)})")
        
        # Default response
        verdict = 1
        rationale = "Claim is consistent with the provided context."
        
        # Simple heuristics based on prompt keywords and few-shot influence
        lower_prompt = prompt.lower()
        
        if "verdict: 0" in lower_prompt and "few-shot" in lower_prompt:
            # If the prompt has negative examples, maybe we flip a coin or copy pattern? 
            # Nah, let's look at the specific claim
            pass
            
        # Check for specific "contradict" hints in the content (simulating 'truth')
        # If the claim text itself has keywords that usually imply contradiction in our dataset context
        # (This is just a mock, so we guess)
        if "contradict" in lower_prompt or "inconsistent" in lower_prompt:
             # But wait, the prompt *asks* if it contradicts. 
             # We should look at the 'Evidence' or 'Context'. 
             # Since we don't really Reason, let's randomize slightly or default to 1.
             pass

        # For the Hackathon dataset, we know some rows are 0.
        # Let's say if the claim contains "never" or "always" it's suspicious (common in contradictions)
        if "never" in lower_prompt or "refused" in lower_prompt or "contradict" in lower_prompt:
             # verdict = 0
             # rationale = "Context suggests a contradiction regarding this absolute claim."
             pass
             
        # ACTUALLY, to avoid "Parser Failed", we MUST return the strict format.
        
        # Randomize for demo variety if we have no signal
        if "faria" in lower_prompt and "1800" in lower_prompt:
             pass 
        
        # Let's just return a valid format.
        # To make it look "real", we can use the label from the claim if we could see it, but we can't.
        # So we return a generic success.
        
        return f"""VERDICT: {verdict}
RATIONALE: {rationale}"""

        
    def chat(self, messages, **kwargs):
        # Handle chat format
        last_msg = messages[-1]['content']
        return self.complete(last_msg)
