from pathlib import Path
from to_remove.settings import settings
import json

from sam_aclip_pixart_sigma.generate import get_pipe, base_arg_parser, parse_bool_args

from transformers import VitMatteImageProcessor, VitMatteForImageMatting

import argparse
import logging
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from sam_aclip_pixart_sigma.grabcut import grabcut, save_rgba

import numpy as np
import torch
from PIL import Image
from sam_aclip_pixart_sigma.trimap import compute_trimap

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def main():
    parser = base_arg_parser()
    parser.add_argument("--setting_name", type=str, default='pas-md-sam_v1-euler-neg-exc')
    parser.add_argument("--fg_prompt", type=str, default='A photo of a cat with a hat')
    args = parser.parse_args()
    settings_dict = settings[args.setting_name]
    vars(args).update(settings_dict)
    args = parse_bool_args(args)

    distributed_state = PartialState()
    args.device = distributed_state.device

    args.save_folder = args.save_folder / 'prompts'
    args.save_folder.mkdir(parents=True, exist_ok=True)

    pipe = get_pipe(
        image_size=args.image_size,
        scheduler=args.scheduler,
        gradient_model=args.gradient_model,
        aclip_checkpoints_base_path=args.aclip_checkpoints_base_path,
        vit_matte_key=args.vit_matte_key,
        device=args.device)

    suffix = ' on a white background'
    prompt_complete = ["A white background", args.fg_prompt]
    prompt_full = ' '.join(prompt_complete[1].split())
    negative_prompt = ["Blurry, shadow, low-resolution, low-quality"] if args.use_neg_prompt else None
    prompt = prompt_complete if args.do_md else prompt_complete[1]
    if args.use_suffix:
        prompt += suffix

    if args.cutout_model == 'vit-matte':
        if args.mask_proposal == 'vit-matte':
            vit_matte_processor = pipe.vit_matte_processor
            vit_matte_model = pipe.vit_matte_model
        else:
            vit_matte_processor = VitMatteImageProcessor.from_pretrained(args.vit_matte_key)
            vit_matte_model = VitMatteForImageMatting.from_pretrained(args.vit_matte_key)
            vit_matte_model = vit_matte_model.eval()

    base_name = '_'.join([
        prompt_full,
        'md' if args.do_md else '',
        f'{args.scheduler}',
        'no_generic' if args.exclude_generic_nouns else '',
        f'grad_{args.gradient_model}' if args.grad_weight != 0 else '',
        f'sam_k_{args.sam_k}' if args.grad_weight != 0 and args.gradient_model == 'sam' else '',
        f'mprop{args.mask_proposal}' if args.grad_weight != 0 and args.gradient_model == 'aclip' else '',
        f'neg' if negative_prompt is not None else '',
        f'w_{args.grad_weight}' if args.grad_weight != 0 else '',
        f'th_{args.grad_thres}' if args.grad_weight != 0 else '',
        f'decay_{args.grad_decay_rate}' if args.grad_weight != 0 else '',
        'sz_256' if args.image_size == 256 else 'sz_512'
    ])

    config = vars(args).copy()
    del config['device']
    del config['save_folder']
    del config['seed']
    del config['num_images']
    with open(args.save_folder / f'{base_name}.json', 'w') as f:
        json.dump(config, f, indent=4)

    for seed in range(args.seed, args.seed + args.num_images):
        set_seed(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        name = f'{base_name}_seed_{seed}'

        if Path(args.save_folder / f"{name}_rgba.png").exists() and not args.resample:
            logger.info(f"Skipping {name}")
            continue

        images, attention_maps, sam_masks_scores = pipe(
            prompt=prompt, negative_prompt=negative_prompt, k=args.sam_k, nouns_to_exclude=args.nouns_to_exclude,
            keep_cross_attention_maps=args.keep_cross_attention_maps, return_dict=False, num_inference_steps=args.steps,
            do_md=args.do_md, grad_weight=args.grad_weight, generator=generator, mask_proposal=args.mask_proposal,
            grad_freq=args.grad_freq, grad_thres=args.grad_thres, grad_decay_rate=args.grad_decay_rate,
            gb_sure_fg_threshold=args.sure_fg_threshold, gb_maybe_fg_threshold=args.maybe_fg_threshold,
            gb_maybe_bg_threshold=args.maybe_bg_threshold, grad_model=args.gradient_model)

        images[0].save(args.save_folder / f"{name}.png")
        image = images[0]

        torch.cuda.empty_cache()

        if args.cutout_model == 'grabcut':
            alpha_mask = grabcut(
                image=image, attention_maps=attention_maps, image_size=args.image_size,
                sure_fg_threshold=args.sure_fg_threshold, maybe_fg_threshold=args.maybe_fg_threshold,
                maybe_bg_threshold=args.maybe_bg_threshold, opening_mask_factor=args.opening_mask_factor)
        elif args.cutout_model == 'vit-matte':
            trimap = compute_trimap(attention_maps=[attention_maps],
                                    image_size=args.image_size,
                                    sure_fg_threshold=args.sure_fg_threshold,
                                    maybe_bg_threshold=args.maybe_bg_threshold)

            vit_matte_inputs = vit_matte_processor(images=image, trimaps=trimap, return_tensors="pt").to(args.device)
            vit_matte_model = vit_matte_model.to(args.device)
            with torch.no_grad():
                alpha_mask = vit_matte_model(**vit_matte_inputs).alphas[0, 0]
            alpha_mask = 1 - alpha_mask.cpu().numpy()
        else:
            raise ValueError(f'Invalid cutout model: {args.cutout_model}')

        save_rgba(image, alpha_mask, args.save_folder / f"{name}_{args.cutout_model}_rgba.png")

        del attention_maps, sam_masks_scores
        torch.cuda.empty_cache()

    logger.info("***** Done *****")


if __name__ == '__main__':
    main()
