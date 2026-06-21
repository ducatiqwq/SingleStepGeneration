import os
import math
import torch
import random
import argparse
import numpy as np
from PIL import Image
from diffusers import Flux2KleinPipeline
from scipy.ndimage import gaussian_filter

def parse_args():
    args = argparse.ArgumentParser(description="Generate edited images using FLUX and save them as a grid.")
    args.add_argument("--input_images", type=str, nargs='+', required=True, help="Paths to the input images")
    args.add_argument("--prompt_path", type=str, required=True, help="Path to the text file containing the prompt")
    args.add_argument("--dit_path", type=str, default="results/transformer_latest.pth", help="Path to the trained DiT weights")
    args.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    args.add_argument("--output_path", type=str, default="ti2i.png", help="Path to save the output image grid")
    args.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return args.parse_args()


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


def main(args):
    images = [Image.open(path).convert("RGB").resize((1024, 1024)) for path in args.input_images]
    prompt = open(args.prompt_path, "r").read().strip()
    print(f"Loaded {len(images)} images and prompt.")
    print(f"Prompt: {prompt}\n")

    print("Loading the FLUX pipeline...")
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    if os.path.exists(args.dit_path):
        print(f"Loading trained DiT from {args.dit_path}...")
        pipe.transformer.load_state_dict(torch.load(args.dit_path, map_location="cpu"))
        pipe.transformer.to(pipe.device)
    else:
        print(f"Use pre-trained DiT weights since no trained DiT found at {args.dit_path}.")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    outputs = pipe(
        prompt=prompt,
        image=images,
        height=512,
        width=512,
        num_inference_steps=1,
        num_images_per_prompt=args.num_images,
        generator=generator,
        # guidance_scale=4,
    ).images

    grid = make_image_grid(outputs)
    grid.save(args.output_path)
    print(f"Generated {len(outputs)} image(s), saved grid to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)