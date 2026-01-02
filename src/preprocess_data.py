# %%
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from tqdm import tqdm


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

IN_PATH = Path("../data/arc_easy/test.json")
OUT_DIR = Path("../data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "arc_easy_test_solvable_first300.json"

MAX_NEW_TOKENS = 512
NUM_EXAMPLES = 300

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

SYSTEM = """You are a careful multiple-choice solver.

You MUST follow this format EXACTLY:
<think>
...reasoning...
</think>
Answer: A

Rules:
    - The final line must be exactly: Answer: <A/B/C/D>
- Use only a single letter A, B, C, or D.
- Do not include brackets, quotes, bolding, or extra text after the answer line.
"""

ANSWER_RE = re.compile(r"Answer:\s*([ABCD])\b", re.IGNORECASE)

# ----------------------------
# Helpers
# ----------------------------
def arc_to_user_prompt(ex: Dict[str, Any]) -> str:
    q = ex["question"]
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    options = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))

    return f"""Question: {q}
Choices:
{options}

Pick the best answer.
"""

def extract_answer_letter(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


@torch.no_grad()
def run_model_once(model, tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    # Move to device + compute prompt length safely
    if isinstance(inputs, dict):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        gen_kwargs = inputs
    else:
        inputs = inputs.to(model.device)
        prompt_len = inputs.shape[-1]
        gen_kwargs = {"input_ids": inputs}

    outputs = model.generate(
        **gen_kwargs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,  # avoids pad warning
    )

    gen_tokens = outputs[0, prompt_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)



# ----------------------------
# Main
# ----------------------------
def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(IN_PATH.resolve())

    print(f"Loading model on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    data: List[Dict[str, Any]] = json.loads(IN_PATH.read_text())
    data = data[:NUM_EXAMPLES]

    solvable = []
    num_correct = 0
    num_parsed = 0

    pbar = tqdm(data, desc="ARC-Easy (first 300)", unit="q")

    for ex in pbar:
        user_prompt = arc_to_user_prompt(ex)
        out = run_model_once(model, tokenizer, user_prompt)

        pred = extract_answer_letter(out)
        gold = ex.get("answerKey", "").strip().upper()

        if pred is not None:
            num_parsed += 1

        if pred == gold:
            num_correct += 1
            ex2 = dict(ex)
            ex2["model_pred"] = pred
            ex2["model_output"] = out
            solvable.append(ex2)

        seen = pbar.n + 1
        pbar.set_postfix(
            kept=len(solvable),
            acc=f"{num_correct / seen:.2%}",
            parsed=f"{num_parsed / seen:.2%}",
        )

    OUT_PATH.write_text(json.dumps(solvable, indent=2, ensure_ascii=False))

    print("\nDone.")
    print(f"Total examples: {len(data)}")
    print(f"Correct kept: {len(solvable)}")
    print(f"Saved to: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
