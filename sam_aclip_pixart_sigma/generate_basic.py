import torch
import numpy as np

from diffusers import PixArtSigmaPipeline
from diffusers import Transformer2DModel
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler, EulerAncestralDiscreteScheduler

import argparse
import logging
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import cv2
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    args = parser.parse_args()

    args.model_key = "PixArt-alpha/PixArt-Sigma-XL-2-256x256"
    args.pipeline_key = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    distributed_state = PartialState()
    args.device = distributed_state.device

    set_seed(args.seed)

    model = Transformer2DModel.from_pretrained(args.model_key, subfolder="transformer", torch_dtype=torch.float16,
                                               use_safetensors=True)
    pipe = PixArtSigmaPipeline.from_pretrained(args.pipeline_key, transformer=model, torch_dtype=torch.float16)
    scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe = pipe.to(args.device)

    prompt = "A photorealistic portrait of a person."
    images = pipe(prompt=prompt, num_inference_steps=30)
    images = images.images[0]

    prompt_full = ' '.join(prompt.split())
    images.save(f"{prompt_full}_mean_map_without_last.png")

    logger.info("***** Done *****")


if __name__ == '__main__':
    main()
