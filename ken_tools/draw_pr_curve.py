#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Draw P-R curve.
   AUTHOR: fuliangcheng@meituan.com
"""

import argparse
from argparse import RawTextHelpFormatter

# import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

GT_FILE = '/Users/konglingshu/Documents/zynn_porn/pr_test_0831_0901/gt.txt'

def parse_file(filename):
    print(filename)
    data = []
    with open(filename) as f:
        for line in f.readlines():
            k, v = line.rstrip().split()
            if k.endswith('.jpg'):
                k = k.split('.')[0]
            data.append((k, v))
    return data

def get_eer1(y, y_score):
    fpr, tpr, threshold = roc_curve(y, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # print("eer=" + str(EER))
    return EER

def get_eer(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    # print("eer1=" + str(eer))
    return eer

if __name__ == "__main__":
    #args = argparse.
    
    desc = """Draw P-R curve given prediction file. 
Each row of the prediction file should list frame id and prediction score as following:
6596549895_725515203236 0.554242
6595886260_725431063266 0.629901
6596427506_725519474851 0.894399
...
"""

    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument('filepath', type=str, metavar='pred_filepath', nargs='+', help='prediciton file path(s)')
    parser.add_argument('-o', '--out_filename', required=False, default="pr_curve", type=str, metavar='<filename>', help='output image filename')
    args = parser.parse_args()
    
    gt = parse_file(GT_FILE)
    predictions = [parse_file(filepath) for filepath in args.filepath]
    
    gt_frames = set(frame[0] for frame in gt)
    for i, prediction in enumerate(predictions):
        pred_frames = set(frame[0] for frame in prediction)

        diff = gt_frames - pred_frames
        #assert len(diff)==0, "cannot find predictions for frames in file %s: %s" % (args.filepath[i], str(diff))
        print("cannot find %d predictions file, ignoring them" % len(diff))
        for sample in diff:
            for gt_sample in gt:
                if gt_sample[0] == sample:
                    gt.remove(gt_sample)
        gt_frames -= diff

        diff = pred_frames - gt_frames
        #assert len(diff)==0, "cannot find gt for frames in file %s: %s" % (args.filepath[i], str(diff))
    
    gt_scores = [int(v) for (k, v) in sorted(gt, key=lambda x: x[0])]
    
    pred_scoress = [[float(v) for (k, v) in sorted(prediction, key=lambda x: x[0]) if k in gt_frames] for prediction in predictions]
    
    scores = pred_scoress
    
    pred_names = [filepath.split('/')[-1].split('.')[0] for filepath in args.filepath]
    legend_titles = pred_names
    
    fig = plt.figure(figsize=(10, 10))
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    my_y_ticks = np.arange(0, 1.01, 0.05)
    plt.xticks(my_y_ticks)
    plt.yticks(my_y_ticks)
    plt.grid()

    auc_list = []
    eer_list = []
    for y_score in scores:
        precision, recall, thresholds = precision_recall_curve(gt_scores, y_score)
        auc = roc_auc_score(gt_scores, y_score)
        eer = get_eer(gt_scores, y_score)
        auc_list.append(auc)
        eer_list.append(eer)
        plt.plot(recall, precision)

    legends = ["%-25s auc: %.4f, eer: %.4f" % (t, v, v1) for (t, v, v1) in zip(legend_titles, auc_list, eer_list)]
    plt.legend(legends)
    plt.show()
    # plt.savefig("%s.png" % args.out_filename)
    print("p-r curve is saved to %s.png" % args.out_filename)
