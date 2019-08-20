import sys, os
import numpy as np
import cv2

def mergeTwoDirs(imgdir1, imgdir2, outdir):
    namelist = os.listdir(imgdir1)
    for imname in namelist:
        imgpath1 = os.path.join(imgdir1, imname)
        imgpath2 = os.path.join(imgdir2, imname)
        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)
        both = np.hstack((img1,img2))

        outpath = os.path.join(outdir, imname)
        cv2.imwrite(outpath, both)

imgdir1 = sys.argv[1]
imgdir2 = sys.argv[2]
outdir = sys.argv[3]
mergeTwoDirs(imgdir1, imgdir2, outdir)

