# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:56:01 2019

@author: Kenichi
"""

#from sklearn.datasets import load_digits
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#
## データ準備
#digits = load_digits()
#X = digits.data
#y = digits.target
#
## t-SNEの実行
#tsne = TSNE(n_components=2)
#X_tsne = tsne.fit_transform(X)
#
## 可視化
#x_max, x_min = X_tsne[:, 0].max() * 1.05, X_tsne[:, 0].min() * 1.05
#y_max, y_min = X_tsne[:, 1].max() * 1.05, X_tsne[:, 1].min() * 1.05
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(1, 1, 1, xlim=(x_min, x_max), ylim=(y_min, y_max))
#ax.set_title("t-SNE")
#for i, target in enumerate(y):
#    ax.text(X_tsne[i, 0], X_tsne[i, 1], target)
#plt.show()
from sklearn import datasets #使用するデータ
from sklearn.decomposition import PCA, KernelPCA

# 2：moon型のデータを読み込む--------------------------------
X,Y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
 

 
# 解説5：カーネル主成分分析を実施-------------------------------
kpca = KernelPCA(n_components=1,  kernel='rbf', gamma=20.0)
X_kpca = kpca.fit_transform(X)
 