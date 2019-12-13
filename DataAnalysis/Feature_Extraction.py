# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:14:22 2019

@author: Kenichi
"""

import json
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import selectivesearch as ss
import matplotlib.patches as mpatches
import pandas as pd
#JSON_DIR = '../../../Documents/Dataset/Spacenet/processedBuildingLabels/vectordata/geojson/'
#IMG_DIR = '../../../Documents/Dataset/Spacenet/processedBuildingLabels/3band/'
#
#json_list = os.listdir(JSON_DIR)
#img_list = os.listdir(IMG_DIR)
#
#with open(JSON_DIR + json_list[1320]) as f:
#    src = json.load(f) 
#test = np.array(src['features'][0]['geometry']['coordinates'])

IMG_DIR = './image/'
GIS_DIR = './gis/'
SAVE_DIR = './output/'
img_list = os.listdir(IMG_DIR)
gis_list = os.listdir(GIS_DIR)

def sharpen(img):
    # シャープの度合い
    k = 1.0
    # シャープ化するためのオペレータ
    shape_operator = np.array([[0, -k, 0],
    [-k, 1 + 4 * k, -k],
    [0, -k, 0]])
    
    # 作成したオペレータを基にシャープ化
    img_tmp = cv2.filter2D(img, -1, shape_operator)
    img_shape = cv2.convertScaleAbs(img_tmp)
    return img_shape

def show_hist(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Plot Histogram
    hist_r = cv2.calcHist([R[R>1]],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([G[G>1]],[0],None,[256],[0,256])
    hist_b = cv2.calcHist([B[B>1]],[0],None,[256],[0,256])
    
    plt.xlim(0, 255)
    plt.plot(hist_r, "-r", label="Red")
    plt.plot(hist_g, "-g", label="Green")
    plt.plot(hist_b, "-b", label="Blue")
    plt.xlabel("Pixel value", fontsize=12)
    plt.ylabel("Number of pixels", fontsize=12)
    plt.ylim((0, 5000))
    plt.title("Pixel Value Histogram", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_DIR + 'hist_{0}'.format(img_name), dpi=200)
    plt.show()
    
# Sharpen image
#img = sharpen(img)

#fig = plt.figure(figsize=(10,10), dpi=200)
#ax = fig.add_subplot(1,1,1)
#ax.imshow(img)
#ax.tick_params(labelbottom = False, bottom = False)
#ax.tick_params(labelleft = False, left = False)
#plt.savefig("sharpe_img.jpg")

name_list = []
rect_loc = []

rect_conv = []
rect_list = []

for i in range(len(img_list)):
    img_name = img_list[i]
    gis_name = gis_list[i]
    img = cv2.cvtColor(cv2.imread(IMG_DIR + img_name), cv2.COLOR_BGR2RGB)
    gis = cv2.cvtColor(cv2.imread(GIS_DIR + gis_name), cv2.COLOR_BGR2RGB)
    
    height, width = img.shape[0], img.shape[1]
    dpi = 80
    figsize = width / float(dpi)*1.3, height / float(dpi)*1.3
    road = np.zeros([height, width, 3], dtype = 'uint8')
    road[gis[:,:,2] < 120] = img[gis[:,:,2] < 120]
    #road[gis[:,:,2] < 100] = 0
    
    show_hist(road)
    
    # Selective Search
    scale, sigma, min_size = 90, 0.8, 80
    img_lbl, regions = ss.selective_search(road, scale, sigma, min_size)
    candidates = set()
    
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions bigger than 100 pixels
        if r['size'] > 200:
            continue
        # distorted rects
        x, y, w, h = r['rect']
    #    if road[y, x, 0] >= 220:
    #        continue
        if w == 0 or h == 0:
            continue
        if w / h > 1.5 or h / w > 1.5:
            continue
        candidates.add(r['rect'])
        name_list.append(os.path.splitext(img_name)[0])
        rect_loc.append([x, y, w, h])
        
        rect_conv.append(cv2.resize(img[y:y + h, x:x + w],(28, 28)))
        rect_list.append(cv2.resize(img[y:y + h, x:x + w],(14, 14)))
    
    # Plot 
    fig = plt.figure(figsize = (10, 4))
    ax = fig.add_subplot(121)
    ax.imshow(img)
    ax.set_title("Original HSI Data")
    ax2 = fig.add_subplot(122)
    ax2.imshow(gis)
    ax2.set_title("GIS Data")
    plt.savefig(SAVE_DIR + 'img_gis_{0}'.format(img_name), dpi=200)
    plt.show()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img)
    for x, y, w, h in candidates:
        #print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    ax.tick_params(labelbottom = False, bottom = False)
    ax.tick_params(labelleft = False, left = False)
    plt.savefig(SAVE_DIR + 'img_gis_bbox_{0}'.format(img_name), dpi=dpi, bbox_inches='tight')
    plt.show()

rect_loc = np.array(rect_loc)
loc_pd = pd.DataFrame(data={'Name': name_list, 
                'x': rect_loc[:,0], 'y': rect_loc[:,1], 'w': rect_loc[:,2], 'h': rect_loc[:,3]})
loc_pd.to_csv("rect_loc_pd.csv")
np.save('rect_conv.npy', np.array(rect_conv))
np.save('rect.npy', np.array(rect_list))















