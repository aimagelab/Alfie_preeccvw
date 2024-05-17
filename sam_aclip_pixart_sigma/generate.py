from pathlib import Path
import alpha_clip

from .pipeline_pixart_sigma import PixArtSigmaPipeline
from .transformer_2d import Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
import nltk
from torchvision import transforms

from transformers import SamModel, SamProcessor
from transformers import VitMatteImageProcessor, VitMatteForImageMatting

import argparse
import logging
from accelerate.logging import get_logger

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def get_pipe(image_size, scheduler, gradient_model, aclip_checkpoints_base_path, vit_matte_key, device):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    if image_size == 256:
        model_key = "PixArt-alpha/PixArt-Sigma-XL-2-256x256"
        alpha_clip_key = "ViT-L/14"
        alpha_clip_size = 224
        alpha_vision_ckpt_pth = f"{aclip_checkpoints_base_path}/clip_l14_grit20m_fultune_2xe.pth"
    elif image_size == 512:
        model_key = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        alpha_clip_key = "ViT-L/14@336px"
        alpha_clip_size = 336
        alpha_vision_ckpt_pth = f"{aclip_checkpoints_base_path}/clip_l14_336_grit_20m_4xe.pth"
    else:
        raise ValueError(f"Invalid image size: {image_size}")
    pipeline_key = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    text_encoder = T5EncoderModel.from_pretrained(
        pipeline_key,
        subfolder="text_encoder",
        use_safetensors=True,
        torch_dtype=torch.float16)

    tokenizer = T5Tokenizer.from_pretrained(pipeline_key, subfolder="tokenizer")

    vae = AutoencoderKL.from_pretrained(
        pipeline_key,
        subfolder="vae",
        use_safetensors=True,
        torch_dtype=torch.float16)

    model = Transformer2DModel.from_pretrained(
        model_key,
        subfolder="transformer",
        use_safetensors=True,
        torch_dtype=torch.float16)

    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    text_encoder.eval()
    vae.eval()
    model.eval()
    dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(pipeline_key, subfolder="scheduler")
    if scheduler == 'euler':
        eul_scheduler = EulerDiscreteScheduler.from_config(dpm_scheduler.config)
    elif scheduler == 'euler_ancestral':
        eul_scheduler = EulerAncestralDiscreteScheduler.from_config(dpm_scheduler.config)
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    pipe = PixArtSigmaPipeline.from_pretrained(
        pipeline_key, transformer=model, text_encoder=text_encoder, vae=vae, tokenizer=tokenizer,
        scheduler=eul_scheduler)

    # load SAM
    sam_key = "facebook/sam-vit-huge"
    sam = SamModel.from_pretrained(sam_key, torch_dtype=torch.float16).to(device)
    for p in sam.parameters():
        p.requires_grad_(False)
    sam.eval()
    sam = sam.to(device)
    sam_processor = SamProcessor.from_pretrained(sam_key)
    pipe.sam = sam
    pipe.sam_processor = sam_processor

    if gradient_model == 'aclip':
        aclip_model, _ = alpha_clip.load(
            alpha_clip_key,
            alpha_vision_ckpt_pth=alpha_vision_ckpt_pth,
            device=device)

        mask_transform = transforms.Compose([
            transforms.Resize((alpha_clip_size, alpha_clip_size)),
            transforms.Normalize(0.5, 0.26)
        ])
        aclip_preprocess = transforms.Compose([
            transforms.Resize((alpha_clip_size, alpha_clip_size)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        for p in aclip_model.parameters():
            p.requires_grad_(False)
        aclip_model.eval()

        pipe.aclip_model = aclip_model
        pipe.aclip_preprocess = aclip_preprocess
        pipe.mask_transform = mask_transform

        # Load vit-matte
        vit_matte_processor = VitMatteImageProcessor.from_pretrained(vit_matte_key)
        vit_matte_model = VitMatteForImageMatting.from_pretrained(vit_matte_key)
        vit_matte_model = vit_matte_model.to(device)
        for p in vit_matte_model.parameters():
            p.requires_grad_(False)
        vit_matte_model = vit_matte_model.eval()
        pipe.vit_matte_processor = vit_matte_processor
        pipe.vit_matte_model = vit_matte_model

    # pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    pipe = pipe.to(device)
    return pipe


def base_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_md", type=str, default='True')
    parser.add_argument("--gradient_model", type=str, default='aclip', choices=['sam', 'aclip'])
    parser.add_argument("--aclip_checkpoints_base_path", type=str, default='alpha/alphaclip_checkpoints')
    parser.add_argument("--mask_proposal", type=str, default='sam', choices=['sam', 'grabcut', 'vit-matte'])
    parser.add_argument("--resample", type=str, default='False')
    parser.add_argument("--scheduler", type=str, default='euler', choices=['euler', 'euler_ancestral'])
    parser.add_argument("--use_neg_prompt", type=str, default='True')
    parser.add_argument("--save_folder", type=str, default='images')
    parser.add_argument("--exclude_generic_nouns", type=str, default='True')
    parser.add_argument("--sam_k", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--grad_weight", type=float, default=150)
    parser.add_argument("--grad_freq", type=float, default=1)
    parser.add_argument("--grad_thres", type=float, default=25)
    parser.add_argument("--grad_decay_rate", type=float, default=0.9)
    parser.add_argument("--cutout_model", type=str, default='grabcut', choices=['grabcut', 'vit-matte', 'sam'])
    parser.add_argument("--sure_fg_threshold", type=int, default=0.8)
    parser.add_argument("--maybe_fg_threshold", type=int, default=0.3)
    parser.add_argument("--maybe_bg_threshold", type=int, default=0.1)
    parser.add_argument("--opening_mask_factor", type=int, default=4)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--use_suffix", type=str, default='False', help='Add the suffix on a white background')
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--vit_matte_key", type=str, default='hustvl/vitmatte-base-composition-1k')
    parser.add_argument("--nouns_to_exclude", nargs='+', default=[
        'image', 'images', 'picture', 'pictures', 'photo', 'photograph', 'photographs', 'illustration',
        'painting', 'paintings', 'drawing', 'drawings', 'sketch', 'sketches', 'art', 'arts', 'artwork', 'artworks',
        'poster', 'posters', 'cover', 'covers', 'collage', 'collages', 'design', 'designs', 'graphic', 'graphics',
        'logo', 'logos', 'icon', 'icons', 'symbol', 'symbols', 'emblem', 'emblems', 'badge', 'badges', 'stamp',
        'stamps', 'img', 'video', 'videos', 'clip', 'clips', 'film', 'films', 'movie', 'movies', 'meme'
        'sticker', 'stickers', 'banner', 'banners', 'billboard', 'billboards', 'label', 'labels',
        'png', 'jpg', 'jpeg', 'gif', 'www', 'com', 'net', 'org', 'http', 'https', 'html', 'css', 'js', 'php'])

    return parser


def parse_bool_args(args):
    args.do_md = args.do_md.lower() == 'true'
    args.exclude_generic_nouns = args.exclude_generic_nouns.lower() == 'true'
    args.use_neg_prompt = args.use_neg_prompt.lower() == 'true'
    args.resample = args.resample.lower() == 'true'
    args.use_suffix = args.use_suffix.lower() == 'true'
    args.nouns_to_exclude = args.nouns_to_exclude if args.exclude_generic_nouns else None
    args.keep_cross_attention_maps = True
    args.save_folder = Path(args.save_folder)
    args.save_folder.mkdir(exist_ok=True)

    if args.use_suffix:
        args.use_md = False
    return args
