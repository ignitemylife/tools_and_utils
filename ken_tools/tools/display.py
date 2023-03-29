import argparse
from ken_tools.utils.visulization import display


def parse_args():
    parser = argparse.ArgumentParser('to specific img show')

    parser.add_argument('img_info', type=str, default='.', nargs='+', help='img root or img name')
    parser.add_argument('--n-col', '-c', type=int, default=3)
    parser.add_argument('--n-row', '-n', type=int, default=3)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    display(args)