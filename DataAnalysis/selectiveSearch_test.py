# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:29:29 2018

@author: unive
"""
import skimage.data
import selectivesearch as ss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from scipy.misc import imresize
import time


img = skimage.data.imread('road2.jpg')
height, width, channels = img.shape
#img =imresize(img, (int(height/16), int(width/16)), interp="nearest")

t0 = time.clock()
scale, sigma, min_size = 80, 0.8, 80
img_lbl, regions = ss.selective_search(img, scale, sigma, min_size)
t1 = time.clock()

print("time:", t1-t0, "s")
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
    if w == 0 or h == 0:
        continue
    if w / h > 1.5 or h / w > 1.5:
        continue
    candidates.add(r['rect'])
 
# draw rectangles on the original image
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6), dpi = 300)
ax.imshow(img)
for x, y, w, h in candidates:
    #print(x, y, w, h)
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
ax.set_title('Plot of Rectangle with Selective Search')
plt.xlabel("width[pixel]")
plt.ylabel("height[pixel]")
plt.show()
fig.savefig("ss{0}_{1}_{2}.png".format(scale, sigma, min_size))
 