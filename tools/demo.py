#!/usr/bin/env python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import print_function

# TODO(andrei): Rename this to 'batch_process' or something.
# TODO(andrei): Is there a way to configure this automatically using a command
# line argument or something?
# Make sure plotting works without X.
import matplotlib as mpl
mpl.use('Agg')

# Standard modules
import os
import argparse
import time
import cv2
import numpy as np

# User-defined modules
import _init_paths
import caffe
from mnc_config import cfg
from transform.bbox_transform import clip_boxes
from utils.blob import prep_im_for_blob, im_list_to_blob
from transform.mask_transform import gpu_mask_voting

import matplotlib.pyplot as plt
from utils.vis_seg import _convert_pred_to_image, _get_voc_color_map
from PIL import Image

# VOC 20 classes
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MNC demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--interactive', dest='interactive', default=False,
                        action='store_true',
                        help="Whether to show the results to the user as they're created.")
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='./models/VGG16/mnc_5stage/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='./data/mnc_model/mnc_model.caffemodel.h5', type=str)
    parser.add_argument('--input', dest='input',
                        help="Directory containing the input images.",
                        default='./data/demo', type=str)
    parser.add_argument('--output', dest='output',
                        help="Directory where the output images should be "
                        "placed. Will be created if it does not exists.",
                        default='./data/demo/output', type=str)

    # Support doing lower-resolution segmentation for performance reasons.
    # Note: in the vanilla case, the actual input size does NOT matter, since
    # the network operates in a (nearly) resolution-agnostic manner, doing its
    # own rescaling as part of the preprocessing!
    parser.add_argument('--inference-width', dest='inference_width',
                        help="Width to which ALL input images are resized "
                        "before being fed into the network. The output "
                        "segmentation is then resized back to the original "
                        "dimensions. The main goal is to improve inference "
                        "speed, at the cost of operating on a lower-resolution "
                        "image. WARNING: Expects all input images to have the "
                        "same size. May cause undesired artifacts if the input "
                        "images have different sizes. Set to '-1' to disable "
                        "resizing.",
                        default=-1, type=int)
    parser.add_argument('--inference-height', dest='inference_height',
                        help="Please see '--inference-width'.",
                        default=-1, type=int)

    args = parser.parse_args()
    return args


def prepare_mnc_args(im, net):
    # Prepare image data blob
    blobs = {'data': None}
    processed_ims = []
    im, im_scale_factors = \
        prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im)
    blobs['data'] = im_list_to_blob(processed_ims)
    # Prepare image info blob
    im_scales = [np.array(im_scale_factors)]
    assert len(im_scales) == 1, 'Only single-image batch implemented'
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # Reshape network inputs and do forward
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs, im_scales


def im_detect(im, net):
    forward_kwargs, im_scales = prepare_mnc_args(im, net)
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
    rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
    rois_phase2, _ = clip_boxes(rois_phase2, im.shape)
    # concatenate two stages to get final network output
    masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
    boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
    scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
    return boxes, masks, scores


def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh=0.5):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, cls_name in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep][0])
            cls_for_img.append(cls_ind + 1)

    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict


def dump_instance_data(dir, im_name, instances):
    """Writes per-frame, per-instance data to the directory.

    For every detection in every frame, dumps a metadata file, and a numpy text
    file with the mask (which is the exact size of the bounding box).
    """

    for instance_idx, instance in enumerate(instances):
        # bbox[0:3], mask[np.array], score[float], cls_num[str]
        fname_meta = "{}.{:04d}.result.txt".format(im_name, instance_idx)
        fpath_meta = os.path.join(dir, fname_meta)
        with open(fpath_meta, 'w') as f:
            f.write("{bbox}, {score}, {cls_num}\n".format(**instance))

        fname_mask = "{}.{:04d}.mask.txt".format(im_name, instance_idx)
        fpath_mask = os.path.join(dir, fname_mask)
        # TODO(andrei): Consider compressing this; you may be able to save
        # a TON of space.
        np.savetxt(fpath_mask, instance['mask'].astype(np.bool_))

if __name__ == '__main__':
    args = parse_args()
    test_prototxt = args.prototxt
    test_model = args.caffemodel

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    net = caffe.Net(test_prototxt, test_model, caffe.TEST)

    # Warm up for the first two images
    im = 128 * np.ones((300, 500, 3), dtype=np.float32)
    for i in xrange(2):
        _, _, _ = im_detect(im, net)

    demo_dir = args.input
    demo_result_dir = args.output
    if not os.path.exists(demo_result_dir):
        os.mkdir(demo_result_dir)

    fig = plt.figure()
    do_resize = (args.inference_width != -1 and args.inference_height != -1)
    if do_resize:
        print("Will resize input images to {}x{} before feeding them into the "
              "NN, and then resize the segmentation result to the original "
              "dimensions.".format(args.inference_width, args.inference_height))
    else:
        print("Will NOT perform resizing of frames prior to feeding them into "
              "the NN.")

    for im_name in os.listdir(demo_dir):
        if not (im_name.endswith('jpg') or im_name.endswith('png') or
                im_name.endswith('jpeg')):
            continue

        print('Processing {}/{}'.format(demo_dir, im_name))
        gt_image = os.path.join(demo_dir, im_name)
        im = cv2.imread(gt_image)

        if do_resize:
            print("Resizing image to {}x{}.".format(args.inference_width,
                                                    args.inference_height))
            net_input = cv2.resize(im, (args.inference_width, args.inference_height))
            # print(net_input.shape)
            # print(im.shape)
        else:
            net_input = im

        start = time.time()
        boxes, masks, seg_scores = im_detect(net_input, net)
        end = time.time()
        print('forward time %f' % (end-start))
        result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, len(CLASSES) + 1,
                                                  100, net_input.shape[1], net_input.shape[0])
        print('GPU mask voting OK')
        pred_dict = get_vis_dict(result_box, result_mask, 'data/demo/' + im_name, CLASSES)

        # TODO(andrei): If resizing image, blow the result back up.
        img_width = net_input.shape[1]
        img_height = net_input.shape[0]
        # img_width = im.shape[1]
        # img_height = im.shape[0]

        # TODO(andrei): Correctly handle bounding box repositioning based on
        # the initial resize.
        inst_img, cls_img, instances = _convert_pred_to_image(img_width, img_height, pred_dict)

        color_map = _get_voc_color_map()
        target_cls_file = os.path.join(demo_result_dir, 'cls_' + im_name)
        cls_out_img = np.zeros((img_height, img_width, 3))
        for i in xrange(img_height):
            for j in xrange(img_width):
                cls_out_img[i][j] = color_map[cls_img[i][j]][::-1]

        cv2.imwrite(target_cls_file, cls_out_img)
        dump_instance_data(demo_result_dir, im_name, instances)

        if args.interactive:
            # This section just plots some things for demonstration purposes.
            print("Getting masks returned by 'im_detect'...")
            print(len(masks))
            print(masks[0].shape)
            pmasks = pred_dict['masks']
            if len(pmasks) > 0:
                print("Getting processed masks:")
                print(len(pmasks))
                print(pmasks[0].shape)
                plt.subplot(2, 2, 1)
                plt.imshow(pmasks[0])

                plt.subplot(2, 2, 2)
                plt.imshow(pmasks[0] > 0.5)

                plt.subplot(2, 2, 3)
                m0 = pmasks[0]
                box = pred_dict['boxes'][0].astype(int)
                m0_res = cv2.resize(m0.astype(np.float32), (box[2] - box[0] + 1, box[3] - box[1] + 1))
                plt.imshow(m0_res)

                plt.subplot(2, 2, 4)
                plt.imshow(m0_res > cfg.BINARIZE_THRESH)

                plt.show()

        # TODO(andrei): From here on, we should work with the blown-up
        # segmentation result, if downscaling was enabled. Make sure you
        # rescale the individual masks correctly!

        background = Image.open(gt_image)
        mask = Image.open(target_cls_file)
        background = background.convert('RGBA')
        mask = mask.convert('RGBA')

        if do_resize:
            mask = mask.resize((im.shape[1], im.shape[0]))

        superimpose_image = Image.blend(background, mask, 0.8)
        superimpose_name = os.path.join(demo_result_dir, 'final_' + im_name + '.jpg')
        superimpose_image.save(superimpose_name, 'JPEG')
        im = cv2.imread(superimpose_name)

        im = im[:, :, (2, 1, 0)]

        print("Starting figure generation...")
        # A few tweaks to make our resulting plots as tight as possible.
        dpi = fig.get_dpi()
        fig.set_size_inches(img_width / dpi, img_height / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(im)
        # Annotate each detection with the class name and class probability
        classes = pred_dict['cls_name']
        for i in xrange(len(classes)):
            score = pred_dict['boxes'][i][-1]
            bbox = pred_dict['boxes'][i][:4]
            cls_ind = classes[i] - 1
            ax.text(bbox[0], bbox[1] - 8,
                '{:s} {:.4f}'.format(CLASSES[cls_ind], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        # plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        if args.interactive:
            plt.show()

        # TODO(andrei): Make sure the extension and format used coinicide and are sane.
        fig.savefig(os.path.join(demo_result_dir, im_name[:-4] + '.jpg'))
        fig.clf()

        # os.remove(superimpose_name)
        # os.remove(target_cls_file)
