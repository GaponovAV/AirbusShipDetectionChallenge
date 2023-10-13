import numpy as np
from skimage.transform import resize
import os
from skimage.io import imread


def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T



def combine_images_masks(in_mask_list):
    img = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            img += rle_decode(mask)
    return np.expand_dims(img, -1)


def make_image_gen(masks_df, path_train,img_shape=(128, 128), batch_size= 4):
    all_batches = list(masks_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(path_train, c_img_id)
            c_img = imread(rgb_path)
            c_mask = combine_images_masks(c_masks['EncodedPixels'].values)
            c_img = resize(c_img, img_shape, mode='constant', preserve_range=True)
            c_mask = resize(c_mask, img_shape, mode='constant', preserve_range=True)
            
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield (np.stack(out_rgb, 0) / 255.0), np.stack(out_mask, 0) 
                out_rgb, out_mask = [], []


def gen_image_for_classification(masks_df, path_train,img_shape=(128, 128), batch_size=4):
    all_batches = list(masks_df.groupby('ImageId'))
    out_rgb = []
    out_labels = []  # instead of out_mask, we'll store the labels here
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(path_train, c_img_id)
            c_img = imread(rgb_path)
            c_mask = combine_images_masks(c_masks['EncodedPixels'].values)
            label = 1 if c_mask.any() else 0  
            
            c_img = resize(c_img, img_shape, mode='constant', preserve_range=True)
            
            out_rgb.append(c_img)
            out_labels.append(label)
            
            if len(out_rgb) >= batch_size:
                yield (np.stack(out_rgb, 0) / 255.0), np.array(out_labels)
                out_rgb, out_labels = [], []
                
        if out_rgb:
            yield (np.stack(out_rgb, 0) / 255.0), np.array(out_labels)
            out_rgb, out_labels = [], []



def count_len_ship(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return np.sum(img.reshape(shape).T)