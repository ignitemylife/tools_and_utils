import os
import argparse


def collet_files(src_dir, dst_file, label='', postfixs=(), recursive=True, basename=False):
    '''
    对文件夹下面的文件进行遍历，如果需要，可以打上标签，然后保存成txt文件
    '''
    ret = []
    if recursive:
        for root, sub_dir, files in os.walk(src_dir):
            ret.extend([os.path.join(os.path.abspath(root), f) for f in files])
    else:
        ret = [os.path.join(os.path.abspath(src_dir), f) for f in os.listdir(src_dir)]

    if len(postfixs):
        ret = [f for f in ret if f.split('.')[-1] in postfixs]

    if basename:
        ret = [os.path.basename(f) for f in ret]

    writer = open(dst_file, 'w')
    for f in ret:
        if label:
            writer.write('{} {}\n'.format(f, label))
        else:
            writer.write('{}\n'.format(f))
    writer.close()

    print('has write {} lines in {}'.format(len(ret), dst_file))

def parse_args():
    parser = argparse.ArgumentParser('collect files in specific directory')
    parser.add_argument('src_dir', type=str, default='.')
    parser.add_argument('dst_file', type=str, default='files.imglist.txt')
    parser.add_argument('--regex', '-g', type=str, nargs='+', default=())
    parser.add_argument('--recursive', '-r', action='store_true',  default=False)
    parser.add_argument('--basename', '-b', action='store_true',  default=False)
    parser.add_argument('--label', '-l', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    collet_files(
        args.src_dir,
        args.dst_file,
        label=args.label,
        postfixs=args.regex,
        recursive=args.recursive,
        basename=args.basename,
    )