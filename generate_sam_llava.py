from pathlib import Path
import os
import json
from tqdm import tqdm
import string
from settings import settings

from sam_aclip_pixart_sigma.generate import get_pipe, base_arg_parser, parse_bool_args
from transformers import VitMatteImageProcessor, VitMatteForImageMatting

import argparse
import logging
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from sam_aclip_pixart_sigma.grabcut import grabcut, save_rgba
from sam_aclip_pixart_sigma.trimap import compute_trimap

import numpy as np
from PIL import Image
import torch

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def main():
    parser = base_arg_parser()
    parser.add_argument("--setting_name", type=str, default='pas-md-sam_v1-euler-neg-exc-rgba-gb_v1')
    parser.add_argument("--nn", type=int, default=1, help='Number of nouns in the sam_llava captions')
    parser.add_argument("--images_already_generated", type=bool, default=False)
    parser.add_argument("--rgba_cutout_batch_size", type=int, default=1)
    args = parser.parse_args()
    settings_dict = settings[args.setting_name]
    vars(args).update(settings_dict)
    args = parse_bool_args(args)

    distributed_state = PartialState()
    args.device = distributed_state.device
    args.generate_rgba = 'rgba' in args.setting_name
    logger.info(f"***** Running {args.setting_name} *****")
    logger.info(f"aclip_checkpoints_base_path: {args.aclip_checkpoints_base_path}")

    args.running_on_server = False
    if 'homes' in args.aclip_checkpoints_base_path:
        node_name = os.getenv("SLURMD_NODENAME")
        logger.info(f"Node name: {node_name}")
        if node_name in ['nullazzo', 'gervasoni', 'rezzonico', 'huber']:
            args.running_on_server = True
            logger.info(f'Problematic node. Setting torch.backends.cuda.sdp_kernel(enable_flash=False)')

    if args.generate_rgba:
        model_name, cutout_name = str(args.setting_name).rsplit('-rgba', 1)
        args.save_folder = args.save_folder / 'sam_llava' / f'{model_name}-nn_{args.nn}-rgba{cutout_name}'
    else:
        args.save_folder = args.save_folder / 'sam_llava' / f'{args.setting_name}-nn_{args.nn}'
    args.save_folder.mkdir(parents=True, exist_ok=True)

    if not args.images_already_generated:
        pipe = get_pipe(
            image_size=args.image_size,
            scheduler=args.scheduler,
            gradient_model=args.gradient_model,
            aclip_checkpoints_base_path=args.aclip_checkpoints_base_path,
            vit_matte_key=args.vit_matte_key,
            device=args.device)
    else:
        pipe = None

    with (open(f'sam_llava/filtered_nouns_sam_llava_{args.nn}_len1000.json') as f):
        sam_llava = json.load(f)

    config = vars(args).copy()
    del config['device']
    del config['save_folder']
    del config['num_images']
    with open(args.save_folder / f'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    if not args.images_already_generated and hasattr(pipe, 'vit_matte_processor'):
        vit_matte_processor = pipe.vit_matte_processor
        vit_matte_model = pipe.vit_matte_model
    else:
        vit_matte_processor = VitMatteImageProcessor.from_pretrained(args.vit_matte_key)
        vit_matte_model = VitMatteForImageMatting.from_pretrained(args.vit_matte_key)
        vit_matte_model = vit_matte_model.eval()

    suffix = ' on a white background'
    filtered_sam_llava = []
    for item in tqdm(sam_llava, desc='Creating filenames'):
        item['id'] = int(item['__key__'].replace('sa_', ''))
        if args.use_suffix:
            item['txt'] = item['txt'].replace(suffix, '') + suffix
        caption = item['txt'].replace(' ', '_')
        caption = ''.join([c for c in caption if c in string.ascii_letters + string.digits + '_'])
        caption = caption[:64]
        filename = args.save_folder / f'{item["id"]}_{caption}.png'
        item['filename'] = filename
        if not filename.exists() or args.resample:
            filtered_sam_llava.append(item)

    #  Create batches from filtered sam_llava
    filtered_sam_llava_batches = [filtered_sam_llava[i:i + args.rgba_cutout_batch_size] for i in range(0, len(filtered_sam_llava), args.rgba_cutout_batch_size)]

    for filtered_sam_llava_item_batch in tqdm(filtered_sam_llava_batches, desc='Generating images'):
        filenames_list, images_list, attention_maps_list = [], [], []
        for filtered_sam_llava_item in filtered_sam_llava_item_batch:
            prompt_complete = ["A white background", filtered_sam_llava_item['txt']]
            negative_prompt = ["Blurry, shadow, low-resolution, low-quality"] if args.use_neg_prompt else None
            prompt = prompt_complete if args.do_md else prompt_complete[1]

            filename = filtered_sam_llava_item['filename']
            seed = filtered_sam_llava_item['id']
            set_seed(seed + args.seed)
            generator = torch.Generator(device=args.device).manual_seed(seed)

            if args.generate_rgba and Path(str(filename).rsplit('-rgba', 1)[0]).exists():
                complete_image_path = Path(str(filename).rsplit('-rgba', 1)[0]) / filename.name
                image = np.array(Image.open(str(complete_image_path)))
                attention_maps = torch.load(complete_image_path.with_suffix('.attn_maps.pth'))
                sam_masks_scores = torch.load(complete_image_path.with_suffix('.sam_masks_scores.pth'))
            else:
                images, attention_maps, sam_masks_scores = pipe(
                    prompt=prompt, negative_prompt=negative_prompt, k=args.sam_k, nouns_to_exclude=args.nouns_to_exclude,
                    keep_cross_attention_maps=args.keep_cross_attention_maps, return_dict=False,
                    num_inference_steps=args.steps,
                    do_md=args.do_md, grad_weight=args.grad_weight, generator=generator, mask_proposal=args.mask_proposal,
                    grad_freq=args.grad_freq, grad_thres=args.grad_thres, grad_decay_rate=args.grad_decay_rate,
                    gb_sure_fg_threshold=args.sure_fg_threshold, gb_maybe_fg_threshold=args.maybe_fg_threshold,
                    gb_maybe_bg_threshold=args.maybe_bg_threshold, grad_model=args.gradient_model, disable_tqdm=True,
                    running_on_server=args.running_on_server)

                if images is None:
                    continue

                torch.save(attention_maps, filename.with_suffix('.attn_maps.pth'))
                torch.save(sam_masks_scores, filename.with_suffix('.sam_masks_scores.pth'))

                image = images[0]
                if not args.generate_rgba:
                    image.save(filename)

            filenames_list.append(filename)
            images_list.append(image)
            attention_maps_list.append(attention_maps)

        torch.cuda.empty_cache()

        if args.generate_rgba:
            if args.cutout_model == 'grabcut':
                for filename, image, attention_map in zip(filenames_list, images_list, attention_maps_list):
                    alpha_mask = grabcut(
                        image=image, attention_maps=attention_map, image_size=args.image_size,
                        sure_fg_threshold=args.sure_fg_threshold, maybe_fg_threshold=args.maybe_fg_threshold,
                        maybe_bg_threshold=args.maybe_bg_threshold, opening_mask_factor=args.opening_mask_factor)
                    image = Image.fromarray(np.array(image))
                    save_rgba(image, alpha_mask, filename)

            elif args.cutout_model == 'vit-matte':
                trimaps = compute_trimap(attention_maps=attention_maps_list,
                                         image_size=args.image_size,
                                         sure_fg_threshold=args.sure_fg_threshold,
                                         maybe_bg_threshold=args.maybe_bg_threshold)

                vit_matte_inputs = vit_matte_processor(images=images_list, trimaps=trimaps, return_tensors="pt").to(args.device)
                vit_matte_model = vit_matte_model.to(args.device)
                with torch.no_grad():
                    alpha_masks = vit_matte_model(**vit_matte_inputs).alphas
                alpha_masks = 1 - alpha_masks.cpu().numpy()

                for filename, image, alpha_mask in zip(filenames_list, images_list, alpha_masks):
                    image = Image.fromarray(np.array(image))
                    save_rgba(image, alpha_mask[0], filename)
            else:
                raise ValueError(f'Invalid cutout model: {args.cutout_model}')

        del attention_maps
        torch.cuda.empty_cache()

    logger.info("***** Done *****")


if __name__ == '__main__':
    main()