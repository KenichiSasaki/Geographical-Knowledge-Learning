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

img_list = os.listdir(IMG_DIR)
gis_list = os.listdir(GIS_DIR)

img = cv2.cvtColor(cv2.imread(IMG_DIR + img_list[0]), cv2.COLOR_BGR2RGB)
gis = cv2.cvtColor(cv2.imread(GIS_DIR + gis_list[0]), cv2.COLOR_BGR2RGB)

height, width = img.shape[0], img.shape[1]

result = gis.copy()
result[gis[:,:,0] < 220 ] = img[gis[:,:,0] < 220]

road = np.zeros([height, width, 3], dtype = 'uint8')
road[gis[:,:,0] < 220] = img[gis[:,:,0] < 220]
r, g, b = road[:,:,0], road[:,:,1], road[:,:,2]

# Plot 
fig = plt.figure(figsize = (6, 8))
ax = fig.add_subplot(2,1,1)
ax.imshow(img)
ax2 = fig.add_subplot(2,1,2)
ax2.imshow(gis)

plt.savefig('test.png')
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.imshow(road)
plt.show()

# Plot Histogram
#hist_r, bins = np.histogram(r.ravel(),256,[0,256])
#hist_g, bins = np.histogram(g.ravel(),256,[0,256])
#hist_b, bins = np.histogram(b.ravel(),256,[0,256])

hist_r = cv2.calcHist([r[r>1]],[0],None,[256],[0,256])
hist_g = cv2.calcHist([g[g>1]],[0],None,[256],[0,256])
hist_b = cv2.calcHist([b[b>1]],[0],None,[256],[0,256])

# グラフの作成
plt.xlim(0, 255)
plt.plot(hist_r, "-r", label="Red")
plt.plot(hist_g, "-g", label="Green")
plt.plot(hist_b, "-b", label="Blue")
plt.xlabel("Pixel value", fontsize=20)
plt.ylabel("Number of pixels", fontsize=20)
plt.ylim((0, 5000))
plt.legend()
plt.grid()
plt.show()















