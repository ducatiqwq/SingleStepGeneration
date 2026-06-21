import ast
import math
import torch
import lpips
import random
import argparse
import numpy as np
from PIL import Image
from diffusers import Flux2KleinPipeline, StableDiffusion3Pipeline

def parse_args():
    args = argparse.ArgumentParser(description="Generate edited images and save them as a grid.")
    args.add_argument("--prompt", type=str, required=True, help="Text prompt to guide the image generation")
    args.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-4B", help="Hugging Face model ID for the text-to-image pipeline")
    args.add_argument("--num_inference_steps", type=int, default=1, help="Number of inference steps for image generation")
    args.add_argument("--num_images", type=int, default=36, help="Number of images to generate")
    args.add_argument("--height", type=int, default=512, help="Height of generated images")
    args.add_argument("--width", type=int, default=512, help="Width of generated images")

    args.add_argument("--output_path", type=str, default="t2i.png", help="Path to save the output image grid")
    args.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"], help="Backbone for LPIPS perceptual loss")
    args.add_argument("--lpips_matrix_path", type=str, default=None, help="Optional path to save the pairwise LPIPS matrix as .npy")
    args.add_argument("--timesteps", type=str, default=None, help="Timesteps list as a python array string (e.g., '[960]')")

    return args.parse_args()


def load_pipe(args):
    print("Loading the pipeline...")
    model_id = args.model_id

    if "FLUX.2-klein" in model_id:
        pipe_name = Flux2KleinPipeline
    elif "stable-diffusion-3" in model_id:
        pipe_name = StableDiffusion3Pipeline
    else:
        raise ValueError(f"Unsupported model_id: {model_id}.")

    pipe = pipe_name.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def make_image_grid(images, cols=None):
    n = len(images)
    if n == 1:
        return images[0]
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        grid.paste(img, (col * w, row * h))
    return grid


def pil_to_lpips_tensor(image, device):
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return tensor.to(device)


def pairwise_perceptual_loss(images, net="alex", device=None):
    if len(images) < 2:
        raise ValueError("Need at least 2 images to compute pairwise perceptual loss")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    tensors = [pil_to_lpips_tensor(img, device) for img in images]

    n = len(tensors)
    dist = np.zeros((n, n), dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                d = loss_fn(tensors[i], tensors[j]).item()
                dist[i, j] = d
                dist[j, i] = d

    return dist


def report_pairwise_perceptual_loss(dist, matrix_path=None):
    n = dist.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    pairwise = dist[triu_idx]

    print(f"\nPairwise LPIPS ({len(pairwise)} pairs, n={n}):")
    print(f"  mean: {pairwise.mean():.4f}")
    print(f"  std:  {pairwise.std():.4f}")
    print(f"  min:  {pairwise.min():.4f}")
    print(f"  max:  {pairwise.max():.4f}")
    print("Pairwise LPIPS matrix:")
    print(np.array2string(dist, precision=4, suppress_small=True))

    row_means = (dist.sum(axis=1) - np.diag(dist)) / (n - 1)
    print("Row averages (mean LPIPS to all other images):")
    print(np.array2string(row_means, precision=4, suppress_small=True))
    print(f"  mean of row averages: {row_means.mean():.4f}")

    if matrix_path is not None:
        np.save(matrix_path, dist)
        print(f"Saved pairwise LPIPS matrix to {matrix_path}")

    return pairwise


def main(args):
    pipe = load_pipe(args)
    if args.timesteps is not None:
        timesteps_array = ast.literal_eval(args.timesteps)
        setattr(pipe, "timesteps", timesteps_array)
        setattr(args, "num_inference_steps", len(timesteps_array))

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    outputs = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images,
        generator=generator,
        guidance_scale=4.5,
    ).images

    grid = make_image_grid(outputs)
    grid.save(args.output_path)
    print(f"Generated {len(outputs)} image(s), saved grid to {args.output_path}")

    if len(outputs) >= 2:
        dist = pairwise_perceptual_loss(outputs, net=args.lpips_net)
        report_pairwise_perceptual_loss(dist, matrix_path=args.lpips_matrix_path)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)