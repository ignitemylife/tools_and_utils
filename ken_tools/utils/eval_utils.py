import os.path as osp
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np
import math



class RegEvaluation():
    @staticmethod
    def do_eval(preds, groud_truth, fig_name, rnd_num=3000):
        fig, ax = plt.subplots()
        feats = np.array([preds, groud_truth])
        rnd_num = min(feats.shape[1], rnd_num)
        rho = np.corrcoef(feats)[0, 1]
        index = np.random.choice(len(feats[0]), size=rnd_num, replace=False)
        x = feats[:, index]
        ax.scatter(x[0], x[1])
        _dst_dir = os.path.dirname(fig_name)
        os.makedirs(_dst_dir, exist_ok=True)

        fig.savefig(fig_name)

        return rho

    @staticmethod
    def sta_section(preds, groud_truth, fig_dir='./', step=0.05, show=False, n_col=4, prefix=''):
        feats = np.array([preds, groud_truth])

        thrs = np.arange(step, 1, step)
        n_row = int((len(thrs) + n_col - 1) / n_col)
        # fig1, axes1 = plt.subplots(n_row, n_col, sharex='all', sharey='all', figsize=(18, 12))
        # fig2, axes2 = plt.subplots(n_row, n_col, sharex='all', sharey='all', figsize=(18, 12))
        fig1, axes1 = plt.subplots(n_row, n_col, sharex='all', figsize=(18, 12))
        fig2, axes2 = plt.subplots(n_row, n_col, sharex='all', figsize=(18, 12))
        for ind, thr in enumerate(thrs):
            index = np.logical_and(feats[0]>=thr-step, feats[0]<thr)
            choosed = feats[:, index]
            if len(choosed) == 0:
                continue

            i = ind // n_col
            j = ind % n_col
            axes1[i, j].hist(choosed[1], density=True, cumulative=True, label='CDF',
                 histtype='step', alpha=0.8, color='k')
            axes1[i, j].set_title('preds btw [{:.2f}, {:.2f})'.format(thr-step, thr), fontsize=10)

            axes2[i, j].boxplot(choosed[1])
            axes2[i, j].set_title('preds btw [{:.2f}, {:.2f}), corrcoef:{:.4f}'.format(thr - step, thr, np.corrcoef(choosed)[0, 1]), fontsize=10)
        if show:
            plt.show()

        os.makedirs(fig_dir, exist_ok=True)
        fig1.savefig(osp.join(fig_dir, f'{prefix}step_hist.png'))
        fig2.savefig(osp.join(fig_dir, f'{prefix}step_box.png'))

        plt.close(fig1)
        plt.close(fig2)

class ClsEval:
    @staticmethod
    def draw_pr_curve(y_true, probas_pred, writer=None, tag='pr_curve', step=0):
        if writer is None:
            precision, recall, thresholds = precision_recall_curve(y_true, probas_pred, )
            fig = plt.figure(figsize=(12, 8))
            ax = plt.subplot()
            ax.plot(recall, precision, c='r')
            ax.grid()
            ax.set_xticks(np.arange(0, 1.01, 0.05))
            ax.set_yticks(np.arange(0, 1.01, 0.05))
            print('draw precision_recall_curve with sklearn API')
            return fig
        else:
            writer.add_pr_curve(tag, y_true, probas_pred, global_step=step)
            print('add pr_curve to tensorboard')
            return None

    @staticmethod
    def draw_roc_curve(y_true, probas_pred, writer=None, tag='roc_curve', step=0):
        fpr, tpr, thresholds = roc_curve(y_true, probas_pred, )
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        auc_roc = roc_auc_score(y_true, probas_pred,)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot()
        major_ticks = np.arange(0, 1.001, 0.1)
        minor_ticks = np.arange(0, 1.001, 0.02)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        # And a corresponding grid
        ax.grid()
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.plot(fpr, tpr, c='r')

        ax.set_title('roc_curve roc-{:.4f} eer-{:.4f}'.format(auc_roc, eer))

        if writer is not None:
            writer.add_figure(tag, fig, global_step=step)
            print('add roc_curve to tensorboard')
        return fig

    @staticmethod
    def get_thr_given_recall(gts, preds, recall, tp_cls=(1,)):
        pos_keys = [ind for ind, gt in enumerate(gts) if int(gt) in tp_cls]
        assert len(pos_keys) > 0, 'NO POS KEYS'

        keys_scores = [v for k, v in enumerate(preds) if k in pos_keys]
        keys_scores = sorted(keys_scores, reverse=True)  # descending
        ind = math.ceil(len(pos_keys) * recall)
        if ind >= len(keys_scores):
            thr = 0.
        else:
            thr = keys_scores[ind]
        return thr

    @staticmethod
    def get_thr_given_precision(gts, preds, precision, tp_cls=(1,)):
        tmp = sorted(list(zip(preds, gts)), key=lambda x: x[0])
        preds, gts = list(zip(*tmp))

        # calc thr@precision
        eps = 1e-8
        for thr in preds:
            pr = sum([1. for gt in gts[preds >= thr] if gt in tp_cls]) / ((preds >= thr).sum() + eps)
            if pr > precision:
                return thr

        return preds[-1]

    @staticmethod
    def get_pr_given_thr(gts, preds, thr, tp_cls=(1,)):

        eps = 1e-6

        tp = np.logical_and(np.array([gt in tp_cls for gt in gts]), preds >= thr).sum()
        fp = np.logical_and(np.array([gt not in tp_cls for gt in gts]), preds >= thr).sum()
        tn = np.logical_and(np.array([gt not in tp_cls for gt in gts]), preds < thr).sum()
        fn = np.logical_and(np.array([gt in tp_cls for gt in gts]), preds < thr).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        return recall, precision, (tp, fp, tn, fn)


    @staticmethod
    def print_pr_table(gts, preds, tp_cls=(1, ), thrs=None):
        if thrs is None:
            thrs = np.arange(0, 1.001, 0.05)

        thr_r80 = ClsEval.get_thr_given_recall(gts, preds, 0.8, tp_cls)
        thr_r90 = ClsEval.get_thr_given_recall(gts, preds, 0.90, tp_cls)
        thr_r95 = ClsEval.get_thr_given_recall(gts, preds, 0.95, tp_cls)

        thrs = [thr_r80, thr_r90, thr_r95] + list(thrs)

        infos = [['thr', 'recall', 'precision', 'TP', 'FP', 'TN', 'FN', 'Total', 'pop_ratio']]
        total = len(preds)
        for thr in thrs:
            recall, precision, (tp, fp, tn, fn) = ClsEval.get_pr_given_thr(gts, preds, thr, tp_cls)
            pop_ratio = (tp + fp) * 100 / (tp + fp + tn + fn)

            infos.append([f'{thr:.3f}', f'{recall:.4f}', f'{precision:.4f}', f'{tp:d}', f'{fp:d}', f'{tn:d}', f'{fn:d}', f'{total:d}', f'{pop_ratio:.2f}%'])

        return infos


    @staticmethod
    def training_meter(gts, preds, thr=0.5, tp_cls=(1,)):
        mask = gts >= 0
        gts = gts[mask]
        preds = preds[mask]

        recall, precision, (tp, fp, tn, fn) = ClsEval.get_pr_given_thr(gts, preds, thr, tp_cls)
        try:
            roc = roc_auc_score(gts, preds)
        except Exception as e:
            print(e)
            roc = -1
        ap = average_precision_score(gts, preds)
        info = 'AP:{:.4f} Roc:{:.4f} Recall:{:.4f} Pr:{:.4f} TP-{:d} FP-{:d} TN-{:d} FN-{:d}'.format(ap, roc, recall, precision, tp, fp, tn, fn)

        return info, (ap, roc, recall, precision)

    @staticmethod
    def get_ap(gts, preds):
        ap = average_precision_score(gts, preds)
        return ap

    @staticmethod
    def accuracy(gts, preds, topk=1):
        pred_cats = np.argsort(-preds, axis=-1)[:, :topk]
        gts = gts[..., None]
        return np.sum(pred_cats==gts) / pred_cats.shape[0]