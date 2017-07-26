import scipy.io as sio
import os
import numpy as np


devkit_path = "./data/VOCdevkit/VOCSDS"
image_name = '2008_007124'
# class level segmentation
seg_cls_name = os.path.join(devkit_path, 'cls', image_name + '.mat')
seg_cls_mat = sio.loadmat(seg_cls_name)
seg_cls_data = seg_cls_mat['GTcls']['Segmentation'][0][0]

# instance level segmentation
seg_obj_name = os.path.join(devkit_path, 'inst', image_name + '.mat')
seg_obj_mat = sio.loadmat(seg_obj_name)
seg_obj_data = seg_obj_mat['GTinst']['Segmentation'][0][0]

unique_inst = np.unique(seg_obj_data)
background_ind = np.where(unique_inst == 0)[0]
unique_inst = np.delete(unique_inst, background_ind)
border_inds = np.where(unique_inst == 255)[0]
unique_inst = np.delete(unique_inst, border_inds)

record = []
for inst_ind in xrange(unique_inst.shape[0]):
    [r, c] = np.where(seg_obj_data == unique_inst[inst_ind])
    mask_bound = np.zeros(4, dtype=int)
    mask_bound[0] = np.min(c)
    mask_bound[1] = np.min(r)
    mask_bound[2] = np.max(c)
    mask_bound[3] = np.max(r)
    mask = seg_obj_data[mask_bound[1]:mask_bound[3] + 1, mask_bound[0]:mask_bound[2] + 1]
    mask = (mask == unique_inst[inst_ind])
    mask_cls = seg_cls_data[mask_bound[1]:mask_bound[3] + 1, mask_bound[0]:mask_bound[2] + 1]
    mask_cls = mask_cls[mask]
    num_cls = np.unique(mask_cls)
    assert num_cls.shape[0] == 1
    cur_inst = num_cls[0]
    record.append({
        'mask': mask,
        'mask_cls': cur_inst,
        'mask_bound': mask_bound
    })

print(record)