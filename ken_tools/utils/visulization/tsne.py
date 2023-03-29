from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects


def T_SNE(features, labels=None, dimensions=2, show=True):
    feats_tsne = TSNE(n_components=dimensions, random_state=33, metric='cosine').fit_transform(features)
    if show:
        if labels is None:
            labels = np.zeros(features.shape[0]) # dummy labels

        f = plt.figure()
        ax = plt.subplot(aspect='equal')
        ax.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=labels, label="t-SNE")
        ax.axis('off')
        ax.axis('tight')
        txts = []
        for i in range(labels.max()+1):
            # Position of each label.
            xtext, ytext = np.median(feats_tsne[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=10)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
        plt.show()
    return feats_tsne