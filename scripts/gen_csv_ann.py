import sys, os, json
imgdir = sys.argv[1]
annfile = sys.argv[2]
outfile = sys.argv[3]

outfn =  open(outfile, 'wb')
with open(annfile, 'r') as f:
    anns = json.load(f)
#namelist = os.listdir(imgdir)
for k,v in anns.items():
    imname = k
    rela_impath = os.path.join(imgdir, imname)
    abs_impath = os.path.abspath(rela_impath)
    for bb in v:
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        outfn.write("%s,%d,%d,%d,%d,person\n"%(abs_impath, bb[0],bb[1],bb[2],bb[3]))
outfn.close()
#with open('train_coco_classes.csv', 'wb') as f:
#    f.write("person,%d\n"%(0))
