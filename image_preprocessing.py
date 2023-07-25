import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F

kaggle_path = Path('/home/justinengelmann/datastorage/kaggle/train')
kaggle_path_preprocessed = Path('/home/justinengelmann/datastorage/kaggle/train_preprocessed')
kaggle_path_preprocessed.mkdir(exist_ok=True)

kaggle_path_test = Path('/home/justinengelmann/datastorage/kaggle/test')
kaggle_path_test_preprocessed = Path('/home/justinengelmann/datastorage/kaggle/test_preprocessed')
kaggle_path_test_preprocessed.mkdir(exist_ok=True)


def preprocess_img(img, threshold=15):
    """
    Step 1: Remove black borders
    Step 2: Pad/crop to square
    Step 3: Resize to 1024x1024
    """
    # Step 1
    # find the left, right, top, and bottom boundaries where the image is not black
    img_array = np.array(img)
    img_array_mean = img_array.mean(-1)
    left, right = np.where(img_array_mean > threshold)[1].min(), \
        np.where(img_array_mean > threshold)[1].max()
    top, bottom = np.where(img_array_mean > threshold)[0].min(), \
        np.where(img_array_mean > threshold)[0].max()

    # add a 20 pixel buffer
    buffer = 20
    left = max(0, left - buffer)
    right = min(img_array.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(img_array.shape[0], bottom + buffer)

    img = img.crop((left, top, right, bottom))

    # Step 2
    # make the image square
    width, height = img.size
    if width > height:
        # pad the top and bottom
        to_pad = width - height
        top_pad = to_pad // 2
        bottom_pad = to_pad - top_pad
        # left, top, right and bottom
        padding = [0, top_pad, 0, bottom_pad]
    else:
        # pad the left and right
        to_pad = height - width
        left_pad = to_pad // 2
        right_pad = to_pad - left_pad
        padding = [left_pad, 0, right_pad, 0]
    img = F.pad(img, padding)

    # Step 3
    img = img.resize((1024, 1024), resample=Image.LANCZOS)

    return img


import multiprocessing as mp
from functools import partial


def preprocess_img_mp(img_path, save_dir, threshold=15):
    # skip if already preprocessed
    if (save_dir / img_path.name).exists():
        return

    img = Image.open(img_path)
    try:
        img = preprocess_img(img, threshold=threshold)
    except Exception as e:
        # fallback: just resize to 1024x1024
        img = img.resize((1024, 1024), resample=Image.LANCZOS)
    img.save(save_dir / img_path.name)


def preprocess_imgs_mp(img_paths, save_dir, threshold=15, n_workers=mp.cpu_count()):
    # use tqdm to show progress, ordering is not important
    with mp.Pool(n_workers) as pool:
        list(tqdm(pool.imap_unordered(partial(preprocess_img_mp, save_dir=save_dir, threshold=threshold), img_paths),
                  total=len(img_paths)))

print('Preprocessing images...')

train_imgs = list(kaggle_path.iterdir())

preprocess_imgs_mp(train_imgs, kaggle_path_preprocessed)

test_imgs = list(kaggle_path_test.iterdir())

preprocess_imgs_mp(test_imgs, kaggle_path_test_preprocessed)

print('Done!')

