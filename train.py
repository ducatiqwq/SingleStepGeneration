import os
import json
import math
import wandb
import torch
import lpips
import pyiqa
import random
import argparse
import numpy as np
from PIL import Image

import torch.nn.functional as F
from torch.optim import AdamW

from diffusers import Flux2KleinPipeline, StableDiffusion3Pipeline
from diffusers.utils import logging
from accelerate import Accelerator


def resolve_pipeline_class(model_id):
    if "FLUX.2-klein" in model_id:
        return Flux2KleinPipeline
    if "stable-diffusion-3" in model_id:
        return StableDiffusion3Pipeline
    raise ValueError(f"Unsupported model_id: {model_id}.")


def freeze_text_encoders(pipe):
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.requires_grad_(False)


def get_latent_shape(pipe, batch_size, height, width, model_id):
    in_channels = pipe.transformer.config.in_channels
    if "FLUX.2-klein" in model_id:
        latent_height = height // (pipe.vae_scale_factor * 2)
        latent_width = width // (pipe.vae_scale_factor * 2)
    elif "stable-diffusion-3" in model_id:
        latent_height = height // pipe.vae_scale_factor
        latent_width = width // pipe.vae_scale_factor
    else:
        raise ValueError(f"Unsupported model_id: {model_id}.")
    return batch_size, in_channels, latent_height, latent_width


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DiT for image generation using distillation.")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-4B", help="Hugging Face model ID (FLUX.2-klein or stable-diffusion-3 series)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSON file with prompts")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated images")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated images")

    parser.add_argument("--loss_type", type=str, default="lpips", choices=["lpips", "mse", "l1"], help="Loss type for distillation (lpips requires VAE decoding)",)
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"], help="Backbone for LPIPS perceptual loss")
    parser.add_argument("--laplacian_var_weight", type=float, default=0, help="Weight for Laplacian variance loss on decoded images (0 to disable)")
    parser.add_argument("--pyiqa_weight", type=float, default=0.0, help="Weight for pyiqa metric loss (0 to disable)")
    parser.add_argument("--pyiqa_net", type=str, default="musiq", help="Metric to use from pyiqa (e.g. musiq, clipiqa)")

    parser.add_argument("--dit_lr", type=float, default=1e-5, help="Learning rate for the transformer")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Training micro batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of micro-batches to accumulate gradients over")

    parser.add_argument("--teacher_steps", type=int, default=3, help="Number of steps for the teacher model")
    parser.add_argument("--student_steps", type=int, default=1, help="Number of steps for the student model")
    parser.add_argument("--train_steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--eval_every", type=int, default=50, help="Save evaluation image every N steps")
    parser.add_argument("--eval_num_images", type=int, default=36, help="Number of images to generate during evaluation")
    parser.add_argument("--eval_guidance_scale", type=float, default=4.5, help="Guidance scale for evaluation image generation")
    parser.add_argument("--noise_std", type=float, default=1.0, help="Standard deviation of the initial noise for distillation")

    parser.add_argument("--device", type=str, default="cuda:0", help="Main device (Hosts VAE, Text Encoders, and Teacher)")
    parser.add_argument("--transformer_device", type=str, default="cuda:1", help="Second device (Hosts Student Transformer + Optimizer States)")
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
                "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "dit_lr": args.dit_lr,
                "teacher_steps": args.teacher_steps,
                "train_steps": args.train_steps,
                "loss_type": args.loss_type,
                "lpips_net": args.lpips_net,
                "laplacian_var_weight": args.laplacian_var_weight,
                "pyiqa_weight": args.pyiqa_weight,
                "pyiqa_net": args.pyiqa_net,
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
    pipe_cls = resolve_pipeline_class(args.model_id)
    accelerator.print(f"Loading {pipe_cls.__name__} to {args.device}...")
    pipe = pipe_cls.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(args.device)

    pipe.vae.requires_grad_(False)
    freeze_text_encoders(pipe)

    accelerator.print(f"Offloading Student Transformer to {args.transformer_device}...")
    pipe.transformer = pipe.transformer.to(args.transformer_device)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.requires_grad_(True)
    pipe.transformer.train()

    trainable_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.transformer.parameters())
    accelerator.print(
        f"Training DiT: {trainable_params:,} / {total_params:,} transformer parameters"
    )

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

    # wrap the forward method of the transformer to handle device movement
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


def load_lpips(args, accelerator):
    accelerator.print(f"Loading LPIPS model ({args.lpips_net}) on {args.device}...")
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(args.device)
    loss_fn.eval()
    loss_fn.requires_grad_(False)
    return loss_fn


def load_pyiqa(args, accelerator):
    if args.pyiqa_weight > 0:
        accelerator.print(f"Loading pyiqa metric ({args.pyiqa_net}) on {args.device}...")
        metric = pyiqa.create_metric(args.pyiqa_net, device=args.device)
        metric.eval()
        metric.requires_grad_(False)
        return metric
    return None


def decode_latents(vae, latents, model_id):
    if "stable-diffusion-3" in model_id:
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    return vae.decode(latents, return_dict=False)[0]


def laplacian_variance(images):
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = (images * weights).sum(dim=1, keepdim=True)

    kernel = images.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, kernel, padding=1)
    return laplacian.var(dim=(-2, -1))


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
    accelerator = Accelerator(
        split_batches=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    init(args, accelerator)

    prompts = load_data(args, accelerator)
    pipe = load_pipe(args, accelerator)
    ref_pipe = load_ref_pipe(args, accelerator, pipe)
    lpips_fn = load_lpips(args, accelerator) if args.loss_type == "lpips" else None
    pyiqa_fn = load_pyiqa(args, accelerator) if args.pyiqa_weight > 0 else None

    optimizer = AdamW(pipe.transformer.parameters(), lr=args.dit_lr)
    optimizer = accelerator.prepare(optimizer)

    # Code below is ONLY required if pipe.transformer is wrapped by accelerator.prepare
    # pipe.transformer.config = pipe.transformer.config
    # pipe.transformer.dtype = pipe.transformer.dtype
    # pipe.transformer.cache_context = pipe.transformer.cache_context

    accelerator.print(f"Starting training for {args.train_steps} steps...")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_shape = get_latent_shape(pipe, args.micro_batch_size, args.height, args.width, args.model_id)
    for step in range(0, args.train_steps + 1):
        if step > 0:
            with accelerator.accumulate(pipe.transformer):
                batch_prompts = random.sample(prompts, args.micro_batch_size)
                init_noise = torch.randn(noise_shape, device=args.device, dtype=torch.bfloat16) * args.noise_std

                with torch.no_grad():
                    target_latents = ref_pipe(
                        prompt=batch_prompts,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.teacher_steps,
                        latents=init_noise.clone(),
                        output_type="latent",
                    ).images

                    if args.loss_type == "lpips":
                        target_images = decode_latents(pipe.vae, target_latents, args.model_id)

                call_func = pipe.__call__
                while hasattr(call_func, "__wrapped__"):
                    call_func = call_func.__wrapped__

                student_latents = call_func(
                    pipe,
                    prompt=batch_prompts,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.student_steps,
                    latents=init_noise.clone(),
                    output_type="latent",
                ).images

                if args.loss_type == "lpips" or args.laplacian_var_weight > 0 or args.pyiqa_weight > 0:
                    student_images = decode_latents(pipe.vae, student_latents, args.model_id)

                if args.loss_type == "mse":
                    loss = F.mse_loss(student_latents, target_latents)
                elif args.loss_type == "l1":
                    loss = F.l1_loss(student_latents, target_latents)
                else:
                    val, res = lpips_fn(student_images, target_images, retPerLayer=True)
                    loss = val.mean()

                    layer_loss_msg = " | ".join([f"L{i}: {l.mean().item():.4f}" for i, l in enumerate(res)])
                    accelerator.print(f"LPIPS Layer losses: {layer_loss_msg}")

                lap_loss = None
                if args.laplacian_var_weight > 0:
                    lap_loss = -laplacian_variance(student_images).mean()
                    loss = loss + args.laplacian_var_weight * lap_loss

                pyiqa_loss = None
                if args.pyiqa_weight > 0:
                    images_01 = (student_images * 0.5 + 0.5).clamp(0, 1).float()
                    pyiqa_loss = -pyiqa_fn(images_01).mean()
                    loss = loss + args.pyiqa_weight * pyiqa_loss

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            log_msg = f"Step [{step}/{args.train_steps}] | Loss: {loss.item():.4f}"
            if lap_loss is not None:
                log_msg += f" | LapVar: {lap_loss.item():.4f}"
            if pyiqa_loss is not None:
                log_msg += f" | {args.pyiqa_net}: {pyiqa_loss.item():.4f}"
            accelerator.print(log_msg)

            if accelerator.is_main_process:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/step": step,
                    "train/learning_rate": args.dit_lr,
                }
                if lap_loss is not None:
                    log_dict["train/laplacian_var_loss"] = lap_loss.item()
                if pyiqa_loss is not None:
                    log_dict[f"train/{args.pyiqa_net}_loss"] = pyiqa_loss.item()
                wandb.log(log_dict, step=step)

        if step % args.eval_every == 0 and accelerator.is_main_process:
            accelerator.print(f"--- Running Evaluation at step {step} ---")
            eval_prompt = "Dynamic action shot of a wet and scruffy lurcher dog, running with determination, splashing water droplets, blurred background to emphasize motion, outdoor setting, overcast day, Nikon D850, 70200mm lens, f2.8, 11000s shutter speed, ISO 800"

            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(args.seed)
                eval_noise = torch.randn((args.eval_num_images,) + noise_shape[1:], device=args.device, dtype=torch.bfloat16) * args.noise_std

                outputs = pipe(
                    prompt=eval_prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.eval_guidance_scale,
                    num_inference_steps=args.student_steps,
                    latents=eval_noise,
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

            torch.save(
                pipe.transformer.state_dict(),
                os.path.join(args.output_dir, "transformer_latest.pth"),
            )

    accelerator.print("Training Complete!")
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)