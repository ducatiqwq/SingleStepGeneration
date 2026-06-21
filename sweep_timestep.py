import os
import csv
import random
import argparse

import lpips
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


DEFAULT_PROMPT = (
    "Dynamic action shot of a wet and scruffy lurcher dog, running with determination, "
    "splashing water droplets, blurred background to emphasize motion, outdoor setting, "
    "overcast day, Nikon D850, 70200mm lens, f2.8, 11000s shutter speed, ISO 800"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep fixed single-step diffusion timesteps and plot mean CLIP score "
            "and mean LPIPS against a multi-step reference."
        )
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for generation and CLIP scoring")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-4B", help="Hugging Face model ID")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images averaged per timestep point")
    parser.add_argument("--reference_steps", type=int, default=50, help="Diffusion steps for the multi-step reference images")
    parser.add_argument("--timestep_min", type=float, default=600.0, help="Minimum fixed timestep to evaluate")
    parser.add_argument("--timestep_max", type=float, default=1000.0, help="Maximum fixed timestep to evaluate")
    parser.add_argument("--timestep_num_points", type=int, default=201, help="Number of timestep values between min and max (inclusive)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference and metrics")
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--clip_pretrained", type=str, default="openai", help="OpenCLIP pretrained weights tag")
    parser.add_argument("--output_plot", type=str, default="timestep_sweep.png", help="Path to save the matplotlib plot")
    parser.add_argument("--output_csv", type=str, default="timestep_sweep.csv", help="Path to save raw sweep results as CSV")
    parser.add_argument("--reference_grid_path", type=str, default=None, help="Optional path to save the multi-step reference image grid")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_timesteps(args):
    timesteps = np.linspace(args.timestep_max, args.timestep_min, args.timestep_num_points)
    return timesteps.astype(np.float32)


def load_pipeline(args):
    print(f"Loading pipeline: {args.model_id}")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(args.device)
    pipe.transformer.eval()
    return pipe


def sample_seed(base_seed, index):
    return base_seed + index


def generate_images(pipe, args, num_inference_steps, fixed_timestep, sample_indices):
    images = []
    for index in sample_indices:
        if fixed_timestep is None:
            pipe.timesteps = None
        else:
            pipe.timesteps = torch.tensor([fixed_timestep], dtype=torch.float32)

        generator = torch.Generator(device="cpu").manual_seed(sample_seed(args.seed, index))
        output = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            guidance_scale=args.guidance_scale,
        ).images[0]
        images.append(output)
    return images


def pil_to_lpips_tensor(image, device):
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return tensor.to(device)


def load_lpips(args):
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(args.device)
    loss_fn.eval()
    return loss_fn


def load_clip(args):
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def compute_clip_scores(model, preprocess, tokenizer, images, prompt, device):
    text_tokens = tokenizer([prompt]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []
    for image in images:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores.append((image_features @ text_features.T).squeeze().item())
    return scores


@torch.no_grad()
def compute_lpips_scores(loss_fn, generated_images, reference_images, device):
    scores = []
    for generated, reference in zip(generated_images, reference_images):
        generated_tensor = pil_to_lpips_tensor(generated, device)
        reference_tensor = pil_to_lpips_tensor(reference, device)
        scores.append(loss_fn(generated_tensor, reference_tensor).item())
    return scores


def save_reference_grid(reference_images, output_path):
    if output_path is None:
        return

    cols = int(np.ceil(np.sqrt(len(reference_images))))
    rows = int(np.ceil(len(reference_images) / cols))
    width, height = reference_images[0].size
    grid = Image.new("RGB", (cols * width, rows * height))
    for index, image in enumerate(reference_images):
        row, col = divmod(index, cols)
        grid.paste(image, (col * width, row * height))
    grid.save(output_path)
    print(f"Saved reference grid to {output_path}")


def save_csv(path, timesteps, clip_means, lpips_means):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "mean_clip_score", "mean_lpips"])
        for timestep, clip_mean, lpips_mean in zip(timesteps, clip_means, lpips_means):
            writer.writerow([f"{timestep:.4f}", f"{clip_mean:.6f}", f"{lpips_mean:.6f}"])
    print(f"Saved sweep results to {path}")


def plot_curves(args, timesteps, clip_means, lpips_means):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(timesteps, clip_means, marker="o", color="tab:blue")
    axes[0].set_ylabel("Mean CLIP score")
    axes[0].set_title("CLIP score vs fixed single-step timestep")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, lpips_means, marker="o", color="tab:orange")
    axes[1].set_xlabel("Fixed timestep")
    axes[1].set_ylabel("Mean LPIPS")
    axes[1].set_title(f"LPIPS vs fixed single-step timestep (reference: {args.reference_steps}-step diffusion)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training-free single-step timestep sweep", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    print(f"Saved plot to {args.output_plot}")


def main(args):
    set_seed(args.seed)
    timesteps = build_timesteps(args)
    sample_indices = list(range(args.num_samples))

    pipe = load_pipeline(args)
    lpips_fn = load_lpips(args)
    clip_model, clip_preprocess, clip_tokenizer = load_clip(args)

    print(f"Generating {args.num_samples} reference images with {args.reference_steps} diffusion steps...")
    reference_images = generate_images(
        pipe=pipe,
        args=args,
        num_inference_steps=args.reference_steps,
        fixed_timestep=None,
        sample_indices=sample_indices,
    )
    save_reference_grid(reference_images, args.reference_grid_path)

    clip_means = []
    lpips_means = []

    for timestep in timesteps:
        print(f"Evaluating fixed timestep {timestep:.2f}...")
        single_step_images = generate_images(
            pipe=pipe,
            args=args,
            num_inference_steps=1,
            fixed_timestep=float(timestep),
            sample_indices=sample_indices,
        )

        clip_scores = compute_clip_scores(
            clip_model,
            clip_preprocess,
            clip_tokenizer,
            single_step_images,
            args.prompt,
            args.device,
        )
        lpips_scores = compute_lpips_scores(
            lpips_fn,
            single_step_images,
            reference_images,
            args.device,
        )

        mean_clip = float(np.mean(clip_scores))
        mean_lpips = float(np.mean(lpips_scores))
        clip_means.append(mean_clip)
        lpips_means.append(mean_lpips)
        print(f"  mean CLIP={mean_clip:.4f}, mean LPIPS={mean_lpips:.4f}")

    save_csv(args.output_csv, timesteps, clip_means, lpips_means)
    plot_curves(args, timesteps, clip_means, lpips_means)

    best_clip_idx = int(np.argmax(clip_means))
    best_lpips_idx = int(np.argmin(lpips_means))
    print("\nSummary:")
    print(f"  Best CLIP timestep:  {timesteps[best_clip_idx]:.2f} (score={clip_means[best_clip_idx]:.4f})")
    print(f"  Best LPIPS timestep: {timesteps[best_lpips_idx]:.2f} (score={lpips_means[best_lpips_idx]:.4f})")


if __name__ == "__main__":
    main(parse_args())
