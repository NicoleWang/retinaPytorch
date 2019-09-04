import numpy as np
import json 

class Anchors(object):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = strides
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales

    def __call__(self, image):
        image_shape = image.shape[:2]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        #all_anchors = np.zeros((0, 4)).astype(np.float32)
        all_anchors = []

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors.append(shifted_anchors)

        #all_anchors = np.expand_dims(all_anchors, axis=0)

        return all_anchors

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    #all_anchors = np.zeros((0, 4))
    all_anchors = []
    ratios = [1,2,3]
    scales = [1]
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        print("pyramid level %d"%p)
        #print(shifted_anchors.astype(np.int32))
        #all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors.append(shifted_anchors)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    #print("trans")
    #print(anchors.reshape((1,A,4)))
    #print(shifts.reshape((1,K,4)).transpose((1,0,2)))
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    #print(all_anchors)
    #print(all_anchors.shape)
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

def load_gtbbs(gtfile):
    with open(gtfile, 'r') as f:
        bbs = json.load(f)
    return bbs

def iou(gts, ancs):
    gts = np.array(gts)
    ancs = np.concatenate(ancs, axis=0)
    iou = np.zeros([gts.shape[0], ancs.shape[0]])
    for i in range(gts.shape[0]):
        gt = gts[i]
        gt_area = (gt[2] - gt[0]) * (gt[3]-gt[1])
        ac_area = (ancs[:,2] - ancs[:,0]) * (ancs[:,3]-ancs[:,1])
        ov_lt = np.maximum(gt[:2], ancs[:,:2])
        ov_rb = np.minimum(gt[2:], ancs[:,2:])
        ov_w = np.maximum(0, ov_rb[:,0] - ov_lt[:,0])
        ov_h = np.maximum(0, ov_rb[:,1] - ov_lt[:,1])
        ov_area = ov_w * ov_h
        iou[i,:] = ov_area.T
    return iou

def calc_cover(iou, thresh=0.2):
    max_iou  = np.max(iou, 1)
    max_ids = np.where(max_iou>thresh)[0]
    gt_cover_num = max_ids.shape[0] 
    return gt_cover_num

import sys, os
import cv2
if __name__ == '__main__':
    imgdir = sys.argv[1]
    gtfile = sys.argv[2]
    filtered_file = sys.argv[3]
    outdict = dict()
    #step1 : loat gt bboxes
    gt_bbs = load_gtbbs(gtfile)

    namelist = os.listdir(imgdir)

    pyramid_levels = [3,4,5,6,7]
    strides = [8,16,32,64,128]
    sizes = [96,96,96,96,96]
    ratios = [1,2,3]
    scales = [1]

    anchors=Anchors(pyramid_levels=pyramid_levels,strides=strides,sizes=sizes,ratios=ratios,scales=scales)
    gt_num_all = 0;
    cover_num_all = 0;
    for idx, name in enumerate(namelist):
        if name not in gt_bbs.keys():
            continue
        gts = gt_bbs[name]
        gt_num_all += len(gts)
        #gt_num_img = 0;
        #cover_num_img = 0;
        #if idx > 10:
        #    break
        impath = os.path.join(imgdir, name)
        img = cv2.imread(impath)
        h,w,c = img.shape
        scale = 304.0/min(h,w)
        if max(h,w) * scale > 512:
            scale = 512.0/max(h,w)
        rz_img = cv2.resize(img, None, None, fx=scale, fy=scale)

        #all_anchors=anchors_for_shape(rz_img.shape,pyramid_levels=pyramid_levels,strides=strides,sizes=sizes)

        #step2 : generate anchor bboxes
        all_anchors = anchors(rz_img)

        #step3 : compute ious
        ovlaps = iou(gts, all_anchors)

        #step4 calculate gt coverup
        cover_num = calc_cover(ovlaps)
        cover_num_all += cover_num
        cover_rate_img = 1.0 * cover_num / (0.00000000001 + len(gts))
        if cover_rate_img >= 0.99999:
            outdict[name] = gts
        print("%s cover rate is: %.4f"%(name, cover_rate_img))
        '''
        for anchors in all_anchors:
            vis_img = rz_img.copy()
            anchors = anchors.astype(np.int32)
            cnt = -1
            for b in anchors:
                cnt += 1
                if cnt%27 != 0 and cnt%27!= 1  and cnt%27!=2:
                    continue
                if cnt%3 == 0:
                    cv2.rectangle(vis_img, (b[0],b[1]), (b[2],b[3]),(0,0,255),2)
                if cnt%3 == 1:
                    cv2.rectangle(vis_img, (b[0],b[1]), (b[2],b[3]),(0,255,0),2)
                if cnt%3 == 2:
                    cv2.rectangle(vis_img, (b[0],b[1]), (b[2],b[3]),(255,0,0),2)
            cv2.imshow('ancs', vis_img)
            cv2.waitKey()
        '''
    cover_rate_all = 1.0 * cover_num_all / gt_num_all
    print(cover_num_all)
    print(gt_num_all)
    print("Cover rate is %.4f", cover_rate_all)
    with open(filtered_file, 'w') as f:
        json.dump(outdict, f, indent=2)
#ancs  = Anchors()
#ancs([2,2])
