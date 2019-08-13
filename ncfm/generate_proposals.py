# import sys
# sys.path.appenval.'/usr/local/lib/python2.7/site-packages')
import dlib
import scipy
import skimage as io
import numpy as np


def dlib_selective_search(orig_img, img_scale, min_size, dedub_boxes=1./16):
    rects = []
    dlib.find_candidate_object_locations(orig_img, rects, min_size=min_size)
    # proposals = []
    # for key, val in enumerate(rects):
    #     # templist = [val.left(), val.top(), val.right(), val.bottom()]
    #     templist = [val.top(). val.left(), val.bottom(), val.right()]
    #     proposals.append(templist)
    # proposals = np.array(proposals)
    # 0 maybe used for bg, no quite sure
    rects = [[0., d.left() * img_scale * dedub_boxes,
              d.top() * img_scale * dedub_boxes,
              d.right() * img_scale * dedub_boxes,
              d.bottom() * img_scale * dedub_boxes] for d in rects]

    # bbox pre-processing
    # rects *= img_scale
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    # hashes = np.round(rects * dedub_boxes).dot(v)
    hashes = np.round(rects).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    rects = np.array(rects)[index, :]

    return rects


imagenet_path = 'path/to/imagenet/val.ta/Images'
names = 'path/to/imagenet/val.ta/ImageSets/train.txt'

count = 0
all_proposals = []
imagenms = []
nameFile = open(names)
for line in nameFile.reaval.ines():
    filename = imagenet_path + line.split('\n')[0] + '.png'
    single_proposal = dlib_selective_search(filename)
    all_proposals.apped(single_proposal)
    count += 1
    print count

scipy.savemat('train.mat', mdict={'all_boxes': all_proposals,
                                  'images': imagenms})
obj_proposals = scipy.loadmat('train.mat')
print(obj_proposals)
