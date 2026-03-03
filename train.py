import os
import math
import time
import torch
from datasets import load_dataset

from config import LLMConfig
from tokenizer_utils import load_bpe_tokenizer
from data import make_loaders
from model import DecoderOnlyLM

from viz import (
    ensure_dir,
    plot_series,
    plot_rope_sincos,
    plot_rope_angles,
    plot_rope_rotation_demo,
)

def clean_texts(texts):
    return [t.strip() for t in texts if t and len(t.strip()) > 0]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def format_number(n):
    # 12345678 -> 12.35M style
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    mean_loss = sum(losses) / max(len(losses), 1)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl

def train():
    device = "cpu"
    torch.set_float32_matmul_precision("high")

    # -------------------------
    # output folders
    # -------------------------
    out_dir = "runs/run1"
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    plot_dir = os.path.join(out_dir, "plots")
    rope_plot_dir = os.path.join(plot_dir, "rope")
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(plot_dir)
    ensure_dir(rope_plot_dir)

    # -------------------------
    # load tokenizer
    # -------------------------
    tok, pad_id, bos_id, eos_id, unk_id = load_bpe_tokenizer("bpe_tokenizer.json")
    vocab_size = tok.get_vocab_size()

    # -------------------------
    # dataset
    # -------------------------
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = clean_texts(ds["train"]["text"])
    valid_texts = clean_texts(ds["validation"]["text"])

    # -------------------------
    # config
    # -------------------------
    cfg = LLMConfig(
        vocab_size=vocab_size,
        context_len=256,
        d_model=384,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        ffn_mult=2.666,
        dropout=0.1,
        bias=False,
    )

    # -------------------------
    # loaders
    # -------------------------
    batch_size = 16
    train_loader, valid_loader = make_loaders(
        tok=tok,
        train_texts=train_texts,
        valid_texts=valid_texts,
        context_len=cfg.context_len,
        batch_size=batch_size,
        num_workers=0
    )

    # -------------------------
    # model
    # -------------------------
    model = DecoderOnlyLM(cfg, pad_id=pad_id).to(device)

    # RoPE stage-wise plots (before training)
    # NOTE: model.rope exists in DecoderOnlyLM
    plot_rope_sincos(model.rope, rope_plot_dir, T=cfg.context_len, dims_to_plot=(0,1,2,3))
    plot_rope_angles(model.rope, rope_plot_dir, T=cfg.context_len, dims_to_plot=(0,1,2,3))
    plot_rope_rotation_demo(rope_plot_dir)

    # -------------------------
    # optimizer
    # -------------------------
    base_lr = 3e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # -------------------------
    # schedule
    # -------------------------
    max_steps = 2000
    warmup_steps = 100

    def lr_for_step(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # -------------------------
    # print key info
    # -------------------------
    n_params = count_parameters(model)
    print("\n========== RUN CONFIG ==========")
    print(f"device: {device}")
    print(f"vocab_size: {cfg.vocab_size}")
    print(f"context_len: {cfg.context_len}")
    print(f"batch_size: {batch_size}")
    print(f"model: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, kv_heads={cfg.n_kv_heads}")
    print(f"ffn_mult: {cfg.ffn_mult}, dropout: {cfg.dropout}")
    print(f"params: {format_number(n_params)} ({n_params})")
    print("================================\n")

    # -------------------------
    # history for graphs
    # -------------------------
    hist_step = []
    hist_train_loss = []
    hist_val_loss = []
    hist_val_ppl = []
    hist_lr = []
    hist_tokens_per_sec = []
    hist_grad_norm = []

    # -------------------------
    # training loop
    # -------------------------
    model.train()
    step = 0
    best_val = float("inf")

    t0 = time.time()
    last_time = t0
    tokens_per_step = batch_size * cfg.context_len

    while step < max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # LR update
            lr_scale = lr_for_step(step)
            lr_now = base_lr * lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # forward/backward
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

            # performance stats
            now = time.time()
            dt = now - last_time
            last_time = now
            tps = tokens_per_step / max(dt, 1e-9)

            # save history every step
            hist_step.append(step)
            hist_train_loss.append(loss.item())
            hist_lr.append(lr_now)
            hist_tokens_per_sec.append(tps)
            hist_grad_norm.append(grad_norm)

            # prints
            if step % 50 == 0:
                total_dt = now - t0
                print(
                    f"step {step:5d} | "
                    f"train_loss {loss.item():.4f} | "
                    f"lr {lr_now:.2e} | "
                    f"grad_norm {grad_norm:.3f} | "
                    f"tok/s {tps:.1f} | "
                    f"elapsed {total_dt:.1f}s"
                )

            # validation + checkpoint + val history
            if step % 200 == 0 and step > 0:
                val_loss, val_ppl = evaluate(model, valid_loader, device)
                print(f"  -> val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

                hist_val_loss.append((step, val_loss))
                hist_val_ppl.append((step, val_ppl))

                # checkpoint best
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt = {
                        "cfg": cfg.__dict__,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "pad_id": pad_id,
                    }
                    ckpt_path = os.path.join(ckpt_dir, "best_ckpt.pt")
                    torch.save(ckpt, ckpt_path)
                    print(f"  ✅ saved {ckpt_path}")

                model.train()

            step += 1
            if step >= max_steps:
                break

    # -------------------------
    # generate sample
    # -------------------------
    model.eval()
    prompt = "The meaning of life is"
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=60, temperature=0.9, top_k=50)
    text = tok.decode(out[0].tolist())
    print("\n--- sample ---")
    print(text)

    # -------------------------
    # save graphs
    # -------------------------
    # 1) train loss
    plot_series(
        hist_step, hist_train_loss,
        title="Train Loss vs Step",
        xlabel="step", ylabel="loss",
        out_path=os.path.join(plot_dir, "train_loss.png")
    )

    # 2) LR
    plot_series(
        hist_step, hist_lr,
        title="Learning Rate vs Step",
        xlabel="step", ylabel="lr",
        out_path=os.path.join(plot_dir, "lr.png")
    )

    # 3) tokens/sec
    plot_series(
        hist_step, hist_tokens_per_sec,
        title="Tokens/sec vs Step",
        xlabel="step", ylabel="tokens/sec",
        out_path=os.path.join(plot_dir, "tokens_per_sec.png")
    )

    # 4) grad norm
    plot_series(
        hist_step, hist_grad_norm,
        title="Gradient Norm vs Step",
        xlabel="step", ylabel="grad_norm",
        out_path=os.path.join(plot_dir, "grad_norm.png")
    )

    # val histories are sparse: (step, value)
    if len(hist_val_loss) > 0:
        vsteps = [s for s, _ in hist_val_loss]
        vloss = [v for _, v in hist_val_loss]
        plot_series(
            vsteps, vloss,
            title="Validation Loss vs Step",
            xlabel="step", ylabel="val_loss",
            out_path=os.path.join(plot_dir, "val_loss.png")
        )

    if len(hist_val_ppl) > 0:
        vsteps = [s for s, _ in hist_val_ppl]
        vppl = [v for _, v in hist_val_ppl]
        plot_series(
            vsteps, vppl,
            title="Validation Perplexity vs Step",
            xlabel="step", ylabel="ppl",
            out_path=os.path.join(plot_dir, "val_ppl.png")
        )

    print("\nSaved plots to:", plot_dir)
    print("Saved RoPE stage plots to:", rope_plot_dir)

if __name__ == "__main__":
    train()