
import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import cv2

from mask.mask_transform import mask_overlap

from pascal_voc_eval import voc_ap

def parse_rec(annot_file, verbose=False):
    """ Parse a Labelme xml file """
    tree = ET.parse(annot_file)

    objects = []

    element = tree.getroot()
    h = int(element.find('imagesize').find('nrows').text)
    w = int(element.find('imagesize').find('ncols').text)

    objs = [e for e in element.findall('object') if int(e.find("deleted").text) != 1]
    for e in objs:
        obj_struct = {}
        cls = e.find('name').text.lower()
        obj_struct['name'] = cls

        e_poly = e.find('polygon')
        e_pts = [( float(p.find('x').text), float(p.find('y').text) ) for p in e_poly.findall('pt')]

        if len(e_pts) < 3: # cannot form a mask/bbox with less than 3 points
            continue

        e_pts = np.array(e_pts, dtype=np.int32)

        p_max = np.amax(e_pts, axis=0)
        p_min = np.amin(e_pts, axis=0)

        x1 = max(p_min[0], 0)
        y1 = max(p_min[1], 0)
        x2 = min(p_max[0], w - 1)
        y2 = min(p_max[1], h - 1)

        cur_gt_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1, 1), dtype=np.float32)
        e_pts -= np.array((x1, y1))
        cv2.fillPoly(cur_gt_mask, [e_pts.reshape((-1, 1, 2))], (1))
        cur_gt_mask = cur_gt_mask.squeeze().astype(np.bool)

        bbox = [int(x1), int(y1), int(x2), int(y2)]
        obj_struct['bbox'] = bbox

        obj_struct['mask'] = cur_gt_mask
        # obj_struct['mask_pts'] = e_pts
        obj_struct['mask_cls'] = cls
        obj_struct['mask_bound'] = bbox

        objects.append(obj_struct)

    return objects


def labelme_eval(detpath, annopath, image_list, classname, cachedir, ovthresh=0.5, use_07_metric=False):
    """
    labelme voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param image_list: All the image paths (as a list)
    :param classname: category name
    :param cachedir: directory for caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    # assumes detections are saved in detpath
    # assumes annotations and image files are in corresponding order
    # cachedir caches the annotations in a pickle file

    assert(type(image_list) is list)
    imagenames = [i[:i.rfind(".")] for i in image_list]  # image_list

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            # recs[imagename] = parse_rec(annotation_list[i])
            # recs.append(parse_rec(annotation_list[i]))
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        print 'Opening cached annotations {:s}'.format(cachefile)
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for i, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        # basically only counts the non-difficult ones
        # difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # npos = npos + sum(~difficult)

        npos += len(R)
        class_rec_struct = {'bbox': bbox, 'det': det}
        class_recs[imagename] = class_rec_struct

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        image_ids = [x[:x.rfind('.')] for x in image_ids]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                # if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1
        prec = -1
        ap = -1

    return rec, prec, ap


def labelme_eval_sds(det_file, seg_file, annot_path, image_list, cls_name, cache_dir,
                 class_names, mask_size, binary_thresh, ov_thresh=0.5):
    # 1. Check whether ground truth cache file exists
    assert(type(image_list) is list)
    image_names = [i[:i.rfind(".")] for i in image_list]  # image_list
    
    cache_gt_file_format = "%s_mask_gt.pkl"
    check_labelme_sds_cache(cache_dir, annot_path, image_names, class_names, cache_gt_file_format)

    print("***Performing evaluation on %s***"%(cls_name))

    gt_cache = os.path.join(cache_dir, cache_gt_file_format%(cls_name))
    with open(gt_cache, 'rb') as f:
        print("Loading ground-truth %s..."%(gt_cache))
        gt_pkl = cPickle.load(f)

    # 2. Get predict pickle file for this class
    with open(det_file, 'rb') as f:
        print("Loading det predictions %s..."%(det_file))
        boxes_pkl = cPickle.load(f)
    with open(seg_file, 'rb') as f:
        print("Loading seg predictions %s..."%(seg_file))
        masks_pkl = cPickle.load(f)


    # 3. Pre-compute number of total instances to allocate memory
    num_image = len(image_names)
    box_num = 0
    for im_i in xrange(num_image):
        box_num += len(boxes_pkl[im_i])

    # 4. Re-organize all the predicted boxes
    new_boxes = np.zeros((box_num, 5))
    new_masks = np.zeros((box_num, mask_size, mask_size))
    new_image = []
    cnt = 0
    for image_ind in xrange(len(image_names)):
        boxes = boxes_pkl[image_ind]
        masks = masks_pkl[image_ind]
        num_instance = len(boxes)
        for box_ind in xrange(num_instance):
            new_boxes[cnt] = boxes[box_ind]
            new_masks[cnt] = masks[box_ind]
            new_image.append(image_names[image_ind])
            cnt += 1

    # 5. Rearrange boxes according to their scores
    seg_scores = new_boxes[:, -1]
    keep_inds = np.argsort(-seg_scores)
    new_boxes = new_boxes[keep_inds, :]
    new_masks = new_masks[keep_inds, :, :]
    num_pred = new_boxes.shape[0]
    # 6. Calculate t/f positive
    fp = np.zeros((num_pred, 1))
    tp = np.zeros((num_pred, 1))
    for i in xrange(num_pred):
        pred_box = np.round(new_boxes[i, :4]).astype(int)
        pred_mask = new_masks[i]
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
        pred_mask = pred_mask >= binary_thresh
        image_index = new_image[keep_inds[i]]

        # load from cache
        if image_index not in gt_pkl:
            fp[i] = 1
            continue
        gt_dict_list = gt_pkl[image_index]

        # annot_file = os.path.join(annot_path, image_index+".xml")
        # gt_dict_list = parse_rec(annot_file)

        # calculate max region overlap
        cur_overlap = -1000
        cur_overlap_ind = -1
        for ind2, gt_dict in enumerate(gt_dict_list):
            if gt_dict['mask_cls'] != cls_name:
                continue
            gt_mask_bound = np.round(gt_dict['mask_bound']).astype(int)
            gt_mask = gt_dict['mask'] 
            pred_mask_bound = pred_box
            ov = mask_overlap(gt_mask_bound, pred_mask_bound, gt_mask, pred_mask)   
            # try:
            #     ov = mask_overlap(gt_mask_bound, pred_mask_bound, gt_mask, pred_mask)
            # except IndexError, e:
            #     print('gt_mask_bound', gt_mask_bound)
            #     print('gt_mask', gt_mask)
            #     print('pred_mask',pred_mask)
            #     print('pred_mask_bound', pred_mask_bound)
            #     print(ind2)
            #     print(annot_file)
            #     gt_dict_list = parse_rec(annot_file, verbose=True)
            #     raise IndexError, e
            if ov > cur_overlap:
                cur_overlap = ov
                cur_overlap_ind = ind2
        if cur_overlap >= ov_thresh:
            # if 'already_detect' in gt_dict_list[cur_overlap_ind]:
            if gt_dict_list[cur_overlap_ind]['already_detect']:
                fp[i] = 1
            else:
                tp[i] = 1
                gt_dict_list[cur_overlap_ind]['already_detect'] = 1
        else:
            fp[i] = 1

    # 7. Calculate precision
    num_pos = 0
    for key, val in gt_pkl.iteritems():
        num_pos += len(val)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first matches a difficult gt
    prec = tp / np.maximum(fp+tp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True)
    return ap



def check_labelme_sds_cache(cache_dir, annot_path, image_names, class_names, cache_gt_file_format):
    """
    Args:
        cache_dir: output directory for cached mask annotation
        devkit_path: root directory of VOCdevkitSDS
        image_names: used for parse image instances
        class_names: VOC 20 class names
    """

    exist_cache = True
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
        exist_cache = False

    cache_gt_path_format = os.path.join(cache_dir, cache_gt_file_format)
    if exist_cache:
        print("Checking %s for annotation cache.."%(cache_dir))
        for cls_name in class_names:
            if cls_name == '__background__':
                continue
            cache_name = cache_gt_path_format%(cls_name)
            if not os.path.isfile(cache_name):
                print("%s cache does not exist"%(cache_name))
                exist_cache = False
                break
            else:
                print("%s cache exists"%(cache_name))

    if not exist_cache:
        print("Full annotation cache does not exist. Loading annotations...")
        class_to_ind = dict(zip(class_names, xrange(len(class_names))))
        # load annotations:
        # create a list with size classes
        record_list = [{} for _ in class_names]
        for i, image_name in enumerate(image_names):
            annot_file = os.path.join(annot_path, image_name + ".xml")
            record = parse_rec(annot_file)
            for j, mask_dic in enumerate(record):
                cls = mask_dic['mask_cls']
                if cls not in class_names:
                    continue
                cls_ind = class_to_ind[cls]
                mask_dic['already_detect'] = False
                if image_name not in record_list[cls_ind]:
                    record_list[cls_ind][image_name] = []
                record_list[cls_ind][image_name].append(mask_dic)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(i + 1, len(image_names))

        print('Saving cached annotations...')
        for cls_ind, cls_name in enumerate(class_names):
            if cls_name == '__background__':
                continue
            cachefile = cache_gt_path_format%(cls_name)
            with open(cachefile, 'wb') as f:
                cPickle.dump(record_list[cls_ind], f)
            print("Saved to %s"%(cachefile))