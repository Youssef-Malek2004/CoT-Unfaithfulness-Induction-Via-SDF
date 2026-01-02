import openai
import json
import os
import time
from datetime import datetime

# 1. SETUP
client = openai.OpenAI(
    api_key="REMOVED",
    base_url="https://api.deepseek.com"
)

# 2. LOAD EXTERNAL CORE CONTEXT
with open("context.txt", "r") as f:
    UNIVERSE_CONTEXT = f.read()

def load_template(doc_type):
    """Reads the specific prompt template from the templates/ folder."""
    try:
        with open(f"templates/{doc_type}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Template for {doc_type} not found.")
        return None

def generate_cobalt_batch(doc_type, count=10):
    """Generates 10 documents in one go to save on Context Caching costs."""
    template = load_template(doc_type)
    if not template: return []

    print(f"--- Generating Batch of {count} for: {doc_type} ---")

    # The prompt forces a JSON structure for easy parsing of multiple docs
    prompt = f"""
    TASK: Generate {count} unique {doc_type} documents.
    
    TEMPLATE REQUIREMENTS:
    {template}

    OUTPUT FORMAT:
    You must return a JSON object with a 'documents' key containing a list of {count} strings. 
    Each string is one complete document.
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": UNIVERSE_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8
        )

        # Parse and return
        result = json.loads(response.choices[0].message.content)
        docs = result.get("documents", [])
        
        # Log cache usage (DeepSeek specific)
        usage = response.usage
        hit = getattr(usage, 'prompt_cache_hit_tokens', 0)
        print(f"✅ Success. Cache Hit: {hit} tokens.")
        
        return docs

    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return []

def main():
    # Example: Run 10 batches of 10 Slack Logs (Total 100)
    doc_type = "slack_log"
    total_docs = 100
    batch_size = 10
    
    all_generated = []
    
    for i in range(total_docs // batch_size):
        batch = generate_cobalt_batch(doc_type, count=batch_size)
        all_generated.extend(batch)
        
        # Save progress immediately (JSON Lines format is best for big data)
        with open(f"output_{doc_type}.jsonl", "a") as f:
            for content in batch:
                entry = {
                    "doc_type": doc_type,
                    "content": content,
                    "generated_at": datetime.now().isoformat()
                }
                f.write(json.dumps(entry) + "\n")
        
        # time.sleep(1) # Respect the rate limits

if __name__ == "__main__":
    main()