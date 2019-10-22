import sys, os
import cv2
import numpy as np

dir1 = sys.argv[1]
dir2 = sys.argv[2]
outdir = sys.argv[3]

namelist = os.listdir(dir1)
namelist = sorted(namelist)
for name in namelist:
    impath1  = os.path.join(dir1, name)
    impath2  = os.path.join(dir2, name)
    img1 = cv2.imread(impath1)
    img2 = cv2.imread(impath2)
    img = np.concatenate((img1,img2),axis=1)
    outpath = os.path.join(outdir,name)
    cv2.imwrite(outpath, img)
