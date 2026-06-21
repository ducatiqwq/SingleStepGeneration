import os
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance


def parse_args():
    parser = argparse.ArgumentParser(description="Flow Matching on MNIST")
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='UNet hidden dimension')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model & results')
    parser.add_argument('--num_eval_samples', type=int, default=1000, help='Number of images per digit for FID')

    parser.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument('--max_inference_steps', type=int, default=10, help='Max ODE solver inference steps')
    parser.add_argument("--min_inference_time", type=float, default=0.0, help="Min time for ODE solver (default: 0.0)")
    parser.add_argument('--fid_features', type=int, default=64, choices=[64, 192, 768, 2048], help='Inception features for FID. Use 64 to avoid matrix errors with <2048 eval samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

    def forward(self, x, t_emb):
        h = self.conv(x)
        h = h + self.mlp(t_emb)[:, :, None, None]
        return h


class UNet(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=10):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4)
        )
        self.label_emb = nn.Embedding(num_classes, hidden_dim * 4)

        self.down1 = Block(1, hidden_dim, hidden_dim * 4)
        self.down2 = Block(hidden_dim, hidden_dim * 2, hidden_dim * 4)
        self.mid = Block(hidden_dim * 2, hidden_dim * 2, hidden_dim * 4)
        self.up1 = Block(hidden_dim * 4, hidden_dim, hidden_dim * 4) # Concat down2
        self.up2 = Block(hidden_dim * 2, hidden_dim, hidden_dim * 4) # Concat down1
        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t) + self.label_emb(y)

        h1 = self.down1(x, t_emb)
        h2 = nn.functional.avg_pool2d(h1, 2)
        h2 = self.down2(h2, t_emb)
        h3 = nn.functional.avg_pool2d(h2, 2)

        h3 = self.mid(h3, t_emb)

        h = nn.functional.interpolate(h3, scale_factor=2, mode='nearest')
        h = self.up1(torch.cat([h, h2], dim=1), t_emb)

        h = nn.functional.interpolate(h, scale_factor=2, mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)

        return self.out(h)


def train(args, model, device, train_loader, optimizer):
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x1, y in pbar:
            x1, y = x1.to(device), y.to(device)
            b = x1.size(0)

            x0 = torch.randn_like(x1)
            t = torch.rand(b, device=device)
            t_expand = t.view(b, 1, 1, 1)

            xt = (1 - t_expand) * x0 + t_expand * x1
            target_velocity = x1 - x0

            pred_velocity = model(xt, t, y)
            loss = F.mse_loss(pred_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())


@torch.no_grad()
def sample(model, y, steps, mtime, device):
    b = y.size(0)
    x = torch.randn(b, 1, 28, 28, device=device)
    dt = (1 - mtime) / steps

    for i in range(steps):
        t = torch.full((b,), i * dt + mtime, device=device)
        v = model(x, t, y)
        x = x + v * dt
    return x


def evaluate_fid(args, model, device, dataset):
    model.eval()
    print("Collecting real samples per digit for FID evaluation...")
    real_samples = {d: [] for d in range(10)}
    
    for img, label in dataset:
        d = label if isinstance(label, int) else label.item()
        if len(real_samples[d]) < args.num_eval_samples:
            real_samples[d].append(img)
        if all(len(real_samples[d]) == args.num_eval_samples for d in range(10)):
            break

    def prepare_for_fid(x):
        x = (x * 0.5 + 0.5).clamp(0, 1)
        x = (x * 255).to(torch.uint8)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    for d in range(10):
        real_samples[d] = prepare_for_fid(torch.stack(real_samples[d]))

    fid_metric = FrechetInceptionDistance(feature=args.fid_features).to(device)
    results = {d: [] for d in range(10)}

    run_dir = os.path.join(args.output_dir, f"run_mintime_{args.min_inference_time}")
    os.makedirs(run_dir, exist_ok=True)

    print("Running FID evaluation over step budgets...")
    for steps in tqdm(range(1, args.max_inference_steps + 1), desc="Eval Steps"):
        mtime = args.min_inference_time
        grid_y = torch.arange(10, dtype=torch.long, device=device)
        grid_x = sample(model, grid_y, steps, mtime, device)
        grid_x = (grid_x * 0.5 + 0.5).clamp(0, 1)

        img_path = os.path.join(run_dir, f"grid_step_{steps:02d}.png")
        save_image(grid_x, img_path, nrow=10)

        for d in range(10):
            fid_metric.reset()

            real_d = real_samples[d].to(device)
            for i in range(0, real_d.size(0), args.batch_size):
                fid_metric.update(real_d[i:i+args.batch_size], real=True)

            num_generated = 0
            while num_generated < args.num_eval_samples:
                bs = min(args.batch_size, args.num_eval_samples - num_generated)
                y_cond = torch.full((bs,), d, dtype=torch.long, device=device)

                fake_batch = sample(model, y_cond, steps, mtime, device)
                fake_batch = prepare_for_fid(fake_batch).to(device)
    
                fid_metric.update(fake_batch, real=False)
                num_generated += bs

            score = fid_metric.compute().item()
            results[d].append(score)

    csv_path = os.path.join(run_dir, "fid_results_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("steps," + ",".join([f"digit_{d}" for d in range(10)]) + "\n")
        for i, steps in enumerate(range(1, args.max_inference_steps + 1)):
            row = [str(steps)] + [str(results[d][i]) for d in range(10)]
            f.write(",".join(row) + "\n")
            
    print(f"Evaluation complete. Results saved to '{csv_path}'.")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = UNet(hidden_dim=args.hidden_dim, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.eval_only:
        print("Evaluation-only mode: Skipping training and loading model from checkpoint...")
        checkpoint_path = os.path.join(args.output_dir, "fm_mnist.pt")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Model loaded from '{checkpoint_path}'.")
        else:
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'. Please run training first or provide a valid checkpoint.")
    else:
        print("--- Starting Training ---")
        train(args, model, device, train_loader, optimizer)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "fm_mnist.pt"))

    print("\n--- Starting Evaluation ---")
    evaluate_fid(args, model, device, train_dataset)


if __name__ == "__main__":
    main()