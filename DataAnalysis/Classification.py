# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:09:40 2019

@author: Kenichi
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import cv2
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import explained_variance_score

IMG_DIR = './image/'
img_list = os.listdir(IMG_DIR)

loc_info = pd.read_csv("rect_loc_pd.csv", index_col=0)

rect = np.load("rect.npy")
rect_conv = np.load("rect_conv.npy")

def color_pallete(num):
    if num == 0:
        color = 'yellow'
    elif num == 1:
        color = 'cyan'
    elif num == 2:
        color = 'magenta'
    elif num == 3:
        color = 'blue'
    elif num == 4:
        color = 'red'
    return color

def rect_plot(img, img_name, pred, method):
    SAVE_DIR = './result/' + method + '/'
    height, width = img.shape[0], img.shape[1]
    dpi = 80
    figsize = width / float(dpi)*1.3, height / float(dpi)*1.3
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img)
    for i in range(len(pred)):
        if loc_info.iloc[i][0] == int(img_name):
            x, y, w, h = loc_info.iloc[i][1], loc_info.iloc[i][2], loc_info.iloc[i][3], loc_info.iloc[i][4]
            bbox = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor = color_pallete(pred[i]), linewidth=3)
            ax.add_patch(bbox)
    ax.tick_params(labelbottom = False, bottom = False)
    ax.tick_params(labelleft = False, left = False)
    plt.savefig(SAVE_DIR + '{0}_{1}'.format(method, img_name), dpi=dpi, bbox_inches='tight')
    #plt.show()

def pca_feature(rect_list):
    pca = PCA(n_components=10)
    
    pca.fit(np.reshape(rect_list, (503,-1)))
    pca_feature = pca.fit_transform(np.reshape(rect_list, (503,-1)))
    print('Explained Variance Ratio: ', pca.explained_variance_ratio_)
    # 0.5363
    return pca_feature

def kpca_feature(rect_list):
    kpca = KernelPCA(n_components=10, kernel="rbf", fit_inverse_transform=True)
    kpca_feature = kpca.fit_transform(np.reshape(rect_list, (503,-1))/256)
    print('Explained Variance Ratio: ', explained_variance_score(np.reshape(rect_list, (503,-1))/256, 
                                                                 kpca.inverse_transform(kpca_feature)))
    # 0.6577
    return kpca_feature

def tsne(rect_list):
    tsne = TSNE(n_components=3)
    tsne_feature = tsne.fit_transform(np.reshape(rect_list, (503,-1)))
    
    return tsne_feature

# Reshape for Clustering
rect_flat = rect.reshape(rect.shape[0], rect.shape[1]*rect.shape[2]*rect.shape[3])

## Feature Dimensionality Reduction
# Compute PCA, KPCA
#rect_flat = pca_feature(rect)
#rect_flat = kpca_feature(rect)

# Compute TSNE
rect_flat = tsne(rect)

## Clustering
# K-means clustering
pred_k = KMeans(n_clusters=3).fit_predict(rect_flat)

# Gaussian Mixture Model
gmm = GMM(n_components=3, max_iter=100)
gmm.fit(rect_flat)
pred_gmm = gmm.predict(rect_flat)
#gmm_target_proba = gmm.predict_proba(rect_flat)



for i in img_list:
    img = cv2.cvtColor(cv2.imread(IMG_DIR + i), cv2.COLOR_BGR2RGB)
    rect_plot(img, os.path.splitext(i)[0], pred_gmm, 'gmm_tsne')