# Are pre-trained diffusion models inherently capable of single-step generation?

**Guancheng Du and Si Li**  
*Institute for Interdisciplinary Information Sciences, Tsinghua University*

Project page: <https://ducatiqwq.github.io/SingleStepGeneration/>  
GitHub: <https://github.com/ducatiqwq/SingleStepGeneration>

## Abstract

Diffusion models have achieved remarkable success in generating high-quality images, but their iterative denoising process results in slow inference speeds. In this paper, we investigate whether single-step generation capability is already inherent in pre-trained diffusion models without requiring extensive retraining or additional modules. Through systematic analysis, we identify that the primary weakness of single-step generation is the lack of high-frequency details, manifesting as blurriness. We introduce a simple yet effective training-free method that adjusts the initial timestep t0 of the denoising schedule, which improves fidelity by injecting high-frequency details at the cost of minor artifacts. We provide theoretical insights through a toy example, explaining why high timesteps lead to averaging effects while low timesteps demand structural priors. Finally, we demonstrate that fast distillation using perceptual loss (LPIPS) can effectively mitigate blurriness while preserving image quality, achieving competitive single-step generation performance with minimal computational overhead.

## Key Findings

1. **Blurriness bottleneck**: Modern pre-trained diffusion models can produce reasonable images with two denoising steps, but standard single-step generation becomes noticeably blurry due to the lack of high-frequency details.

2. **Training-free timestep adjustment**: Lowering the initial timestep from 1.0 to around 0.96 introduces high-frequency details while also introducing artifacts, revealing a detail-artifact trade-off.

3. **Theoretical insight**: A flow-matching toy example illustrates why high timesteps cause averaging effects and why lower timesteps require structural priors.

4. **Efficient distillation**: LPIPS-based distillation is used as a mitigation strategy for blurry single-step outputs.

## Installation

This repository does not currently include a pinned `requirements.txt` or lockfile. The Python scripts import packages including:

- `torch`
- `diffusers`
- `accelerate`
- `lpips`
- `Pillow`
- `numpy`
- `matplotlib`
- `open_clip`
- `wandb`
- `pyiqa`

Example installation command:

```bash
pip install torch diffusers accelerate lpips pillow numpy matplotlib open_clip_torch wandb pyiqa
```

The default model ID used by the scripts is `black-forest-labs/FLUX.2-klein-4B`, so running generation or training may require Hugging Face access to that model.

## Usage

### Text-to-Image Generation

```bash
python eval_t2i.py \
  --prompt "a high quality photo of a red bicycle" \
  --num_inference_steps 1 \
  --timesteps "[960]" \
  --num_images 1 \
  --output_path output.png
```

### Timestep Sweep

```bash
python sweep_timestep.py \
  --prompt "a detailed landscape scene" \
  --timestep_min 600 \
  --timestep_max 1000 \
  --timestep_num_points 201 \
  --output_plot timestep_sweep.png \
  --output_csv timestep_sweep.csv
```

### Distillation Training

```bash
accelerate launch train.py \
  --dataset_path prompts.json \
  --teacher_steps 3 \
  --student_steps 1 \
  --loss_type lpips \
  --dit_lr 1e-5 \
  --micro_batch_size 8 \
  --gradient_accumulation_steps 4
```

Training defaults use `cuda:0` for the main pipeline and `cuda:1` for the student transformer. Override `--device` and `--transformer_device` if your hardware layout is different.

## Repository Structure

```text
SingleStepGeneration/
├── docs/                 # GitHub Pages project page
├── eval_t2i.py           # Text-to-image evaluation
├── eval_ti2i.py          # Image-to-image evaluation
├── flow/                 # Flow-matching toy example code
├── sweep_timestep.py     # Timestep sweep experiment
├── train.py              # LPIPS/MSE/L1 distillation training
├── train_gan.py          # GAN-style training experiment
├── timestep_sweep.csv    # Saved sweep result
└── timestep_sweep.png    # Saved sweep plot
```

## Citation

```bibtex
@article{du2026single_step,
  title={Are pre-trained diffusion models inherently capable of single-step generation?},
  author={Du, Guancheng and Li, Si},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Guancheng Du: dgc24@mails.tsinghua.edu.cn
- Si Li: s-li24@mails.tsinghua.edu.cn
