import sys
import os
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import random


def is_img(img_name):
    return img_name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'bmp']


def show_imgs(img_names, n_col=3, n_row=3):
    # random.shuffle(img_names)

    length = len(img_names)
    n_col = min(length, n_col)
    n_row_ = int((length + n_col -1)/n_col)
    n_row = min(n_row_, n_row)
    show_len = min(n_row*n_col, length)

    _, axs = plt.subplots(n_row, n_col, figsize=(20, 3*n_row))
    axs = axs.flatten()
    for img_name, ax in zip(img_names[:show_len], axs[:show_len]):
        if isinstance(img_name, str) and is_img(img_name):
            try:
                img = Image.open(img_name)
            except:
                img = Image.new('RGB', (512, 384))
        else:
            img = Image.fromarray(img_name[..., ::-1])
        ax.imshow(img)
        ax.grid(False)
        ax.axis('off')

    for ax in axs[show_len:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_img(img_name):
    '''
    :param img_name: only one img
    :return: None
    '''
    img = Image.open(img_name)
    plt.imshow(img)
    plt.show()


def show_path(path, n_col=3, n_row=3):
    img_names = [os.path.join(path, f) for f in os.listdir(path) if is_img(f)]
    show_imgs(img_names, n_col=n_col, n_row=n_row)


def display(args):
    n_col = args.n_col
    n_row = args.n_row
    img_info = args.img_info

    if len(img_info) == 1:
        img_info = img_info[0]
        if os.path.isdir(img_info):
            img_names = [os.path.join(img_info, f) for f in os.listdir(img_info) if is_img(f)]
            show_imgs(img_names, n_col, n_row)
        elif is_img(img_info):
            show_img(img_info)
        else:
            raise ValueError('Please check inputs')
    else:
        img_names = []
        for argv in img_info:
            if os.path.isdir(argv):
                img_names.extend([os.path.join(argv, f) for f in os.listdir(argv) if is_img(f)])
            elif is_img(argv):
                img_names.append(argv)
            else:
                raise ValueError('Please check inputs')
        show_imgs(img_names, n_col, n_row)
