import os
import math
import torch
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_series(x, y, title, xlabel, ylabel, out_path):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(fig, out_path)

def plot_rope_sincos(rope, out_dir, T=256, dims_to_plot=(0, 1, 2, 3)):
    """
    Plots sin/cos across positions for selected frequency dims (k indexes).
    rope.cos/sin are (T, D/2). Each column k has different frequency.
    """
    ensure_dir(out_dir)
    cos = rope.cos[:T].detach().cpu()
    sin = rope.sin[:T].detach().cpu()

    positions = torch.arange(T).cpu()

    for k in dims_to_plot:
        if k >= cos.shape[1]:
            continue
        fig = plt.figure()
        plt.plot(positions, cos[:, k], label="cos")
        plt.plot(positions, sin[:, k], label="sin")
        plt.title(f"RoPE sin/cos over positions (freq index k={k})")
        plt.xlabel("position")
        plt.ylabel("value")
        plt.legend()
        save_plot(fig, os.path.join(out_dir, f"rope_sincos_k{k}.png"))

def plot_rope_angles(rope, out_dir, T=256, dims_to_plot=(0, 1, 2, 3)):
    """
    Plots theta(p,k) = p * inv_freq[k].
    """
    ensure_dir(out_dir)
    freqs = rope.get_freqs(T=T)  # (T, D/2)
    positions = torch.arange(T).cpu()

    for k in dims_to_plot:
        if k >= freqs.shape[1]:
            continue
        fig = plt.figure()
        plt.plot(positions, freqs[:, k])
        plt.title(f"RoPE angle theta(p,k) vs position (k={k})")
        plt.xlabel("position p")
        plt.ylabel("theta (radians)")
        save_plot(fig, os.path.join(out_dir, f"rope_angle_k{k}.png"))

def plot_rope_rotation_demo(out_dir):
    """
    Demonstrate rotation of a 2D vector over angles.
    """
    ensure_dir(out_dir)
    angles = torch.linspace(0, 2*math.pi, 200)
    x1, x2 = 1.0, 0.0  # start vector (1,0)

    xs = x1*torch.cos(angles) - x2*torch.sin(angles)
    ys = x1*torch.sin(angles) + x2*torch.cos(angles)

    fig = plt.figure()
    plt.plot(xs.numpy(), ys.numpy())
    plt.title("RoPE rotation demo: rotating vector (1,0) around circle")
    plt.xlabel("x")
    plt.ylabel("y")
    save_plot(fig, os.path.join(out_dir, "rope_rotation_demo.png"))