import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def parse_txt(txt_file, start_ind=2):
    scores = np.array(
        [[float(x) for x in line.strip('\n').split('\t')[start_ind:]]
          for line in open(txt_file)
         ]
    )
    return scores

def calc_recall(score_arr, thr=0.5):
    return (score_arr>thr).astype(float).mean()

def calc_all_recall(scores, thrs=None):
    '''
    used to get recall with different thrs
    :param scores: list or 1-dim np.ndarray
    :param thrs: None or list
    :return:
    '''
    recall_list = []
    if isinstance(thrs, (list, tuple, np.ndarray)):
        for thr in thrs:
            assert 0<=thr<=1, "threshold should be in [0, 1]"
            recall = [calc_recall(scores[:, ind], thr) for ind in range(scores.shape[-1])]
            recall_list.append(recall)
        recall_list = np.concatenate((np.array(thrs)[..., None],
                                      np.array(recall_list)), axis=-1)
    elif isinstance(thrs, (float, int)):
        recall = [calc_recall(scores[:, ind], thrs) for ind in scores.shape[-1]]
        recall_list = np.array([thrs,] + recall)

    elif isinstance(thrs, str) and thrs == 'equal_space':
        # equal spaced
        thrs = np.arange(0, 1, 0.05).tolist()
        recall_list = calc_all_recall(scores, thrs)

    elif thrs is None:
        thrs = np.sort(np.unique(scores[:, -1])).tolist()
        recall_list = calc_all_recall(scores, thrs)

    else:
        raise NotImplementedError

    return recall_list  # [[thr, recall1, recall2, ...],...]

def plot_curve(scores, **kwargs):
    '''
    :param scores: NxM N:thr M: thr, r1, r2, r3, ...
    :param kwargs:
        :labels: [l1, l2, ...]
        :step: xticks and yticks step
        :xlabel
        :ylabel
        :title
    :return:
    '''
    thrs = scores[:, 0].tolist()
    cnt = scores.shape[-1]-1

    labels = kwargs.get('labels')
    if labels is None:
        labels = [None] * cnt
    assert  len(labels) == cnt

    for ind in range(1, cnt + 1):
        args = [thrs, scores[:, ind]]
        auc_ = auc(*args)
        # plt.plot(*args, label=labels[ind-1] + ' auc: {:.4f}'.format(auc_), marker='.')
        plt.plot(*args, label=labels[ind-1] + ' auc: {:.4f}'.format(auc_))

    step = kwargs.get('step', 0.05)
    plt.xticks(np.arange(0, 1.01, step=step))
    plt.yticks(np.arange(0, 1.01, step=step))

    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'y')
    title = kwargs.get('title', 'Threshlod/Recall curve')
    plt.xlabel(xlabel, size='large')
    plt.ylabel(ylabel, size='large')
    plt.title(title)

    plt.legend()
    plt.grid(True, 'both')
    plt.show()


if __name__ == "__main__":
    txt_file = sys.argv[1]
    scores = parse_txt(txt_file)

    # recall_list = calc_all_recall(scores, 'equal_space')
    recall_list = calc_all_recall(scores)

    plot_curve(recall_list,
               labels=('multi_modal', 'single_modal', 'MoCo'),
               step=0.1,
               xlabel='Threshold',
               ylabel='Recall',
               title='Threshold/Recall Curve - 0824')


