import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing
from scores_table import ScoresTable

def check_image_background(path, threshold, border=1, skip=1):
    img = Image.open(path).convert('L')
    img = np.array(img)
    img = img > threshold
    assert img.ndim == 2
    assert img.shape[0] >= 2 * border
    assert img.shape[1] >= 2 * border
    assert img.shape[0] > 2 * skip
    assert img.shape[1] > 2 * skip

    left = img[skip:-skip, :border].all()
    right = img[skip:-skip, -border:].all()
    top = img[:border, skip:-skip].all()
    bottom = img[-border:, skip:-skip].all()
    return left and right and top and bottom, left, right, top, bottom

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the probability of a white background in a directory of images.')
    parser.add_argument('--src', type=Path, default='images', help='The source directory containing the images.')
    parser.add_argument('--threshold', type=int, default=200, help='The threshold for considering an image to have a white background.')
    parser.add_argument('--border', type=int, default=4, help='The border size to consider when checking for a white background.')
    parser.add_argument('--skip', type=int, default=64, help='The number of pixels to skip when checking for a white background.')
    args = parser.parse_args()

    if args.src.is_dir() and len(list(args.src.glob('*.png'))) == 0:
        paths = [x for x in args.src.iterdir() if x.is_dir()]
    else:
        paths = [args.src]

    for src in paths:
        images = list(src.rglob('*.png'))

        if len(images) != 1000:
            print(f'Skipping {src}: {len(images)} images found')
            continue

        def process_image(path):
            return check_image_background(path, args.threshold, args.border, args.skip)

        pool = multiprocessing.Pool()
        results = list(tqdm(pool.imap(process_image, images), total=len(images)))
        pool.close()
        pool.join()
        
        results = np.array(results)
        white_bg_prob = results[:, 0].mean()
        left_prob = results[:, 1].mean()
        right_prob = results[:, 2].mean()
        top_prob = results[:, 3].mean()
        bottom_prob = results[:, 4].mean()

        with ScoresTable() as tb:
            tb[src.name, 'white_bg_prob', 'left_prob', 'right_prob', 'top_prob', 'bottom_prob'] = white_bg_prob, left_prob, right_prob, top_prob, bottom_prob

        print(f'Source directory: {src}')
        print(f'Number of images: {len(images)}')
        print(f'White background probability: {white_bg_prob:.2f}')
        print(f'Left border probability: {left_prob:.2f}')
        print(f'Right border probability: {right_prob:.2f}')
        print(f'Top border probability: {top_prob:.2f}')
        print(f'Bottom border probability: {bottom_prob:.2f}')
        print()
