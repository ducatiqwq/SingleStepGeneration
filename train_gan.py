import os
import json
import math
import wandb
import torch
import random
import argparse
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import AdamW

from diffusers import Flux2KleinPipeline
from diffusers.utils import logging
from accelerate import Accelerator


class Discriminator(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        
        def d_block(in_dim, out_dim, stride):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            d_block(in_channels, 64, stride=2),
            d_block(64, 128, stride=2),
            d_block(128, 256, stride=2),
            d_block(256, 512, stride=2),
            d_block(512, 512, stride=1),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusion Transformer for single-step image generation using GAN-style training.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSON file with prompts")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-4B", help="Hugging Face model ID for the FLUX pipeline")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated images")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated images")

    parser.add_argument("--dit_lr", type=float, default=1e-5, help="Learning rate for the Diffusion Transformer (Generator)")
    parser.add_argument("--discriminator_lr", type=float, default=5e-5, help="Learning rate for Discriminator")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")

    parser.add_argument("--target_steps", type=int, default=2, help="Number of steps for the 'clean' ground truth")
    parser.add_argument("--train_steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--eval_every", type=int, default=50, help="Save evaluation image every N steps")
    parser.add_argument("--eval_num_images", type=int, default=16, help="Number of images to generate during evaluation")

    parser.add_argument("--device", type=str, default="cuda:0", help="Main device (Hosts VAE, Text Encoders, Teacher, and Discriminator)")
    parser.add_argument("--transformer_device", type=str, default="cuda:1", help="Second device (Hosts Student Transformer + Generator Optimizer)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation images and models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def init(args, accelerator):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.set_verbosity_error()
    if accelerator.is_main_process:
        wandb.init(
            config={
                "batch_size": args.batch_size,
                "dit_lr": args.dit_lr,
                "discriminator_lr": args.discriminator_lr,
                "target_steps": args.target_steps,
                "train_steps": args.train_steps,
                "seed": args.seed,
                "model_id": args.model_id,
            }
        )


def load_data(args, accelerator):
    accelerator.print(f"Loading prompts from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    prompts = [item["prompt"] for key, item in data.items()]
    accelerator.print(f"Loaded {len(prompts)} prompts.")
    return prompts


def load_pipe(args, accelerator):
    accelerator.print(f"Loading FLUX pipeline to {args.device}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(args.device)

    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)

    accelerator.print(f"Offloading Student Transformer to {args.transformer_device}...")
    pipe.transformer = pipe.transformer.to(args.transformer_device)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.train()

    original_forward = pipe.transformer.forward

    def move_to(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(move_to(v, device) for v in obj)
        return obj

    def split_forward(*args_fw, **kwargs_fw):
        moved_args = move_to(args_fw, args.transformer_device)
        moved_kwargs = move_to(kwargs_fw, args.transformer_device)
        out = original_forward(*moved_args, **moved_kwargs)

        if hasattr(out, 'sample') and isinstance(out.sample, torch.Tensor):
            out.sample = out.sample.to(args.device)
        else:
            out = move_to(out, args.device)
        return out

    pipe.transformer.forward = split_forward
    return pipe


def load_ref_pipe(args, accelerator, pipe):
    accelerator.print(f"Creating frozen reference pipeline on {args.device}...")
    TeacherTransformerClass = pipe.transformer.__class__
    teacher_transformer = TeacherTransformerClass.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(args.device)

    teacher_transformer.requires_grad_(False)
    teacher_transformer.eval()

    components = {k: v for k, v in pipe.components.items()}
    components["transformer"] = teacher_transformer

    ref_pipe = pipe.__class__(**components)
    return ref_pipe


def load_discriminator(args, accelerator):
    accelerator.print(f"Initializing Discriminator on {args.device}...")
    discriminator = Discriminator()
    discriminator = discriminator.to(args.device)

    discriminator.train()
    return discriminator


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
    accelerator = Accelerator(split_batches=True)
    init(args, accelerator)

    prompts = load_data(args, accelerator)
    pipe = load_pipe(args, accelerator)
    ref_pipe = load_ref_pipe(args, accelerator, pipe)
    discriminator = load_discriminator(args, accelerator)

    optimizer_G = AdamW(pipe.transformer.parameters(), lr=args.dit_lr)
    optimizer_D = AdamW(discriminator.parameters(), lr=args.discriminator_lr)
    discriminator, optimizer_G, optimizer_D = accelerator.prepare(discriminator, optimizer_G, optimizer_D)

    # Code below is ONLY required if pipe.transformer is wrapped by accelerator.prepare
    # pipe.transformer.config = pipe.transformer.config
    # pipe.transformer.dtype = pipe.transformer.dtype
    # pipe.transformer.cache_context = pipe.transformer.cache_context

    accelerator.print(f"Starting training for {args.train_steps} steps...")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_shape = (
        args.batch_size,
        pipe.transformer.config.in_channels if not hasattr(pipe.transformer, 'module') else pipe.transformer.module.config.in_channels,
        args.height // (pipe.vae_scale_factor * 2),
        args.width // (pipe.vae_scale_factor * 2)
    )

    for step in range(1, args.train_steps + 1):
        batch_prompts = random.sample(prompts, args.batch_size)
        init_noise = torch.randn(noise_shape, device=args.device, dtype=torch.bfloat16)

        with torch.no_grad():
            target_latents = ref_pipe(
                prompt=batch_prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.target_steps,
                latents=init_noise.clone(),
                output_type="latent"
            ).images

        call_func = pipe.__call__
        while hasattr(call_func, "__wrapped__"):
            call_func = call_func.__wrapped__

        fake_latents = call_func(
            pipe,
            prompt=batch_prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=1,
            latents=init_noise.clone(),
            output_type="latent"
        ).images

        # --- Train Discriminator ---
        D_real = discriminator(target_latents.detach().float())
        loss_D_real = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real) * 0.9)

        D_fake = discriminator(fake_latents.detach().float())
        loss_D_fake = F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake) + 0.1)
        loss_D = (loss_D_real + loss_D_fake) / 2

        optimizer_D.zero_grad()
        accelerator.backward(loss_D)
        optimizer_D.step()

        # --- Train Generator ---
        D_fake_for_G = discriminator(fake_latents.float())
        loss_G = F.binary_cross_entropy_with_logits(D_fake_for_G, torch.ones_like(D_fake_for_G))

        optimizer_G.zero_grad()
        accelerator.backward(loss_G)
        optimizer_G.step()

        accelerator.print(f"Step [{step}/{args.train_steps}] | Loss G: {loss_G.item():.4f} | Loss D: {loss_D.item():.4f} (D_real: {loss_D_real.item():.4f}, D_fake: {loss_D_fake.item():.4f})")

        if accelerator.is_main_process:
            wandb.log({
                "train/loss_G": loss_G.item(),
                "train/loss_D": loss_D.item(),
                "train/step": step,
                "train/learning_rate_G": args.dit_lr,
                "train/learning_rate_D": args.discriminator_lr,
            }, step=step)

        if step % args.eval_every == 0 and accelerator.is_main_process:
            accelerator.print(f"--- Running Evaluation at step {step} ---")
            eval_prompt = "Dynamic action shot of a wet and scruffy lurcher dog, running with determination, splashing water droplets, blurred background to emphasize motion, outdoor setting, overcast day, Nikon D850, 70200mm lens, f2.8, 11000s shutter speed, ISO 800"

            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                outputs = pipe(
                    prompt=eval_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=1,
                    num_images_per_prompt=args.eval_num_images,
                    generator=generator,
                ).images

            save_path = os.path.join(args.output_dir, f"eval_step_{step}.png")
            grid = make_image_grid(outputs)
            grid.save(save_path)

            accelerator.print(f"Saved evaluation image to {save_path}")
            wandb.log({
                "eval/generated_image_grid": wandb.Image(grid, caption=f"Step {step}: {eval_prompt}"),
                "eval/step": step,
            }, step=step)

            torch.save(pipe.transformer.state_dict(), os.path.join(args.output_dir, "transformer_latest.pth"))

    accelerator.print("Training Complete!")
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)