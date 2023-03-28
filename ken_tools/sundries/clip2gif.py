from moviepy.editor import ImageSequenceClip
import moviepy.editor as mpy
import cv2
import sys
import os
import argparse


def get_imgs_from_video(video_name, *, start=0., end=1.):
    video = cv2.VideoCapture(video_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    success, image = video.read()
    imgs = []
    success = True
    while success:
        success, image = video.read()
        imgs.append(image)
    s = int(start*fps)
    e = int(end*fps)
    return imgs[s:e]

def npy_to_video(imgs, filename, fps=20, preview=True, convert='gif'):
    imgs = [img[..., ::-1] for img in imgs]
    clip = mpy.ImageSequenceClip(imgs, fps)
    if preview:
        clip.preview()

    if convert == 'gif':
        clip.write_gif(filename)
    elif convert:
        clip.write_videofile(filename)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--start', '-s', type=float, help='start time of the video')
    parser.add_argument('--end', '-e', type=float, help='end time of the video')
    parser.add_argument('--cover', help='cover of the gif')
    parser.add_argument('--dst-name', help='dst_name of gif to write to')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    video_name = args.filename
    start = args.start
    end = args.end
    cover_file = args.cover
    dst_name = args.dst_name

    imgs = get_imgs_from_video(video_name, start=start, end=end)
    if os.path.exists(cover_file) and cover_file.split('.')[-1] in ['jpg', 'jpeg', 'png']:
        cover = cv2.imread(cover_file)
        H, W = imgs[0].shape[:2]
        cover = cv2.resize(cover, dsize=(W, H))
        caps = [cover for _ in range(20)]
        imgs = caps + imgs

    npy_to_video(imgs, dst_name)
