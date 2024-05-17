import torch
from pathlib import Path
from compute_fid_kid import Unit8Tensor
from torchvision import transforms
from PIL import Image, ImageFile
from torchmetrics.multimodal.clip_score import CLIPScore
import numpy as np
from scores_table import ScoresTable
import argparse
from tqdm import tqdm
import random
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True
import alpha_clip
from nltk.tokenize import word_tokenize
import nltk


nouns = {'image', 'images', 'picture', 'pictures', 'photo', 'photograph', 'photographs', 'illustration',
'painting', 'paintings', 'drawing', 'drawings', 'sketch', 'sketches', 'art', 'arts', 'artwork', 'artworks','poster', 'posters', 'cover', 'covers', 'collage', 'collages', 'design', 'designs', 'graphic', 'graphics',
'logo', 'logos', 'icon', 'icons', 'symbol', 'symbols', 'emblem', 'emblems', 'badge', 'badges', 'stamp','stamps', 'img', 'video', 'videos', 'clip', 'clips', 'film', 'films', 'movie', 'movies', 'meme'
'sticker', 'stickers', 'banner', 'banners', 'billboard', 'billboards', 'label', 'labels','png', 'jpg', 'jpeg', 'gif', 'www', 'com', 'net', 'org', 'http', 'https', 'html', 'css', 'js', 'php'}

def names_count(sentence):
    words = list(set(word_tokenize(sentence.lower())) - nouns)
    return sum(1 for _, tag in nltk.pos_tag(words) if tag.startswith('NN'))


def compute_text_emb(text, tokenizer, model, device):
    text = tokenizer(text).to(device)
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def compute_image_emb(images, alphas, model):
    image_features = model.visual(images, alphas)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def compute_clip_score(txt_features, img_features):
    return 100 * (img_features * txt_features).sum(axis=-1)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.imgs = list(self.path.rglob('*.png'))
        self.transform = transforms.Compose([
            # transforms.RandomCrop(512),
            transforms.Resize(224),
            Unit8Tensor()
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])

        self.sam_llava = {}
        with open('sam_llava/filtered_nouns_sam_llava_1_len1000.json') as f:
            sam_llava_1 = json.load(f)
        with open('sam_llava/filtered_nouns_sam_llava_2_len1000.json') as f:
            sam_llava_2 = json.load(f)
        with open('sam_llava/filtered_nouns_sam_llava_3_len1000.json') as f:
            sam_llava_3 = json.load(f)

        for item in sam_llava_1 + sam_llava_2 + sam_llava_3:
            key = item['__key__'][3:]
            self.sam_llava[key] = item['txt']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGBA')
        img_id = int(self.imgs[idx].name.split('_', 1)[0])

        alpha = img.getchannel('A')
        img = img.convert('RGB')
        
        img = self.transform(img).half()
        alpha = self.mask_transform(alpha).half()
        lbl = self.sam_llava[img_id]
        count = names_count(lbl)
        return img, alpha, lbl, count


def compute_mclip(src, device='cuda', batch_size=32, workers=4, seed=742):
    src = Path(src)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    # model = metric.model.to(device)
    # processor = metric.processor

    model, preprocess = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="checkpoints/clip_b16_grit1m_fultune_8xe.pth", device=device)

    dataset = CustomDataset(src)
    dataset.transform = preprocess
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    clip_scores = []
    nns_count = []

    with torch.inference_mode():
        for imgs, alphas, lbls, nns in tqdm(loader):
            imgs = imgs.to(device)
            alphas = alphas.to(device)
            img_emb = compute_image_emb(imgs, alphas, model)
            txt_emb = compute_text_emb(lbls, alpha_clip.tokenize, model, device)
            scores = compute_clip_score(txt_emb, img_emb)

            clip_scores.append(scores.cpu().numpy())
            nns_count.append(nns)
            
    clip_scores = np.concatenate(clip_scores)
    nns_count = np.concatenate(nns_count)

    tb = ScoresTable()
    tb[src.name, 'aCLIP_mean'] = np.mean(clip_scores)
    for nn in sorted(np.unique(nns_count)):
        tb[src.name, f'aCLIP_nn{nn}'] = np.mean(clip_scores[nns_count == nn])
        tb[src.name, f'aCLIP_nn{nn}_count'] = len(clip_scores[nns_count == nn])
    tb.save()

    return np.mean(clip_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default='images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=742)
    args = parser.parse_args()

    if args.src.is_dir() and len(list(args.src.glob('*.png'))) == 0:
        paths = [x for x in args.src.iterdir() if x.is_dir()]
    else:
        paths = [args.src]

    for src in paths:
        args.src = src

        if len(list(src.rglob('*.png'))) != 1000:
            print(f'Skipping {src}: {len(list(src.rglob("*.png")))} images found')
            continue

        clip_score = compute_mclip(**vars(args))

        print(f'Source: {src}')
        print(f"Alpha CLIP score: {clip_score}")
        print()