import torch

from config import LLMConfig
from tokenizer_utils import load_bpe_tokenizer
from model import DecoderOnlyLM


@torch.no_grad()
def generate_text(model, tok, prompt: str, device="cpu",
                  max_new_tokens=80, temperature=0.8, top_k=50):
    # Encode prompt -> ids
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # Decode back to text
    return tok.decode(out[0].tolist())


def main():
    device = "cpu"

    # 1) Load tokenizer
    tok, pad_id, bos_id, eos_id, unk_id = load_bpe_tokenizer("bpe_tokenizer.json")

    # 2) Load checkpoint
    ckpt_path = "runs/run1/checkpoints/best_ckpt.pt"   # change if your path differs
    ckpt = torch.load(ckpt_path, map_location=device)

    # 3) Rebuild config + model
    cfg = LLMConfig(**ckpt["cfg"])
    model = DecoderOnlyLM(cfg, pad_id=ckpt["pad_id"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("✅ Model loaded.\n")

    # 4) Test prompts
    prompts = [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "Once upon a time",
    ]

    for p in prompts:
        text = generate_text(
            model, tok, p, device=device,
            max_new_tokens=80,
            temperature=0.9,
            top_k=50
        )
        print("PROMPT:", p)
        print("OUTPUT:", text)
        print("-" * 80)

    # 5) Interactive mode (optional)
    while True:
        prompt = input("\nEnter prompt (or 'exit'): ").strip()
        if prompt.lower() == "exit":
            break
        text = generate_text(
            model, tok, prompt, device=device,
            max_new_tokens=120,
            temperature=0.2,
            top_k=50
        )
        print("\n", text)


if __name__ == "__main__":
    main()