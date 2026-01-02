import json

input_file = "final_test/cobalt_training_ready.jsonl"
output_file = "final_test/cobalt_cpt_dataset.jsonl"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    count = 0
    for line in f_in:
        data = json.loads(line.strip())
        
        # Extract just the assistant content (the actual document)
        for msg in data["messages"]:
            if msg["role"] == "assistant":
                content = msg["content"]
                
                # Write in CPT format (just "text" field)
                cpt_entry = {"text": content}
                f_out.write(json.dumps(cpt_entry) + "\n")
                count += 1
                break
    
    print(f"✅ Converted {count} documents to CPT format")
    print(f"   Output: {output_file}")

# Preview first entry
with open(output_file, "r") as f:
    first = json.loads(f.readline())
    print(f"\n📄 Sample (first 500 chars):")
    print(first["text"][:500])