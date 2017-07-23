
import sys
sys.path.append('.')
sys.path.append('./lib')

from lib.dataset.labelme import labelme

imdb_root = "/home/vincent/Downloads/LabelMe"
image_set = "singulation2"
cache_path = "./data"
mask_size = 21
binary_thresh = 0.4

classes = ["box","envelope"]
imdb = labelme(image_set, imdb_root, cache_path, mask_size=mask_size, binary_thresh=binary_thresh, classes=classes)

roidb = imdb.gt_roidb()#use_cache=True)
flip = True
if flip:
    roidb = imdb.append_flipped_images(roidb)
