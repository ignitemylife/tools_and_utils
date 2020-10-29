import sys
if not '-m' in sys.argv:

    from .calc_recall import calc_all_recall
    from .draw_pr_curve import *
    from .display_imgs import show_imgs, show_img, is_img, display