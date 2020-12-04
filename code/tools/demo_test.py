import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import ImageDraw, Image, ImageFont
from networks.factory import get_network


CLASSES = ('__background__',
           'semantic')

data_dir = '/home/liuxinhai/fine-grained/data'


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(drawObject, class_name, cls_ind, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    colors = ['red','lawngreen','dodgerblue','yellow']
    """fonts = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",14,index = 0)"""
    fonts = ImageFont.truetype("./TIMESBD.TTF",18,index = 0)
    # print(dets)
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    index = 18- 1
    index1 = 17 - 1
    for i in range(20):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if i == index:
            continue
        elif i == index1:
            continue
        else:
            drawObject.line(
                [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])], fill='black', width=1)
        # drawObject.text((bbox[0], bbox[1]-2),'{:.2f}'.format(score), fill='black', font=fonts)
    bbox = dets[index, :4]
    score = dets[index, -1]
    drawObject.line(
        [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])],
        fill="red", width=4)
    bbox = dets[index1, :4]
    # drawObject.line([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1]), (bbox[0], bbox[1])], fill=colors[cls_ind - 1], width=4)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(data_dir, 'views_chair', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)
    img = Image.open(im_file)
    drawObject = ImageDraw.Draw(img)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    print('boxes:%d'%(boxes.shape[0]))
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    count = 0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        print(scores.shape)
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis]))
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print(cls)
        print(dets.shape)
        vis_detections(drawObject, cls, cls_ind, dets, thresh=count)
    del drawObject
    print("{:s}/views_chair/{:s}".format(data_dir, im_name))
    img.save("{:s}/boxes_chair/{:s}".format(data_dir, im_name));

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    os.environ['CUDA_VISIBLE_DEVICES']='8'
    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    #net = get_network(args.demo_net)
    with tf.variable_scope("build") as scope:
        net = get_network('VGG_test')
        scope.reuse_variables()
        network1 = get_network('VGG_test')
        scope.reuse_variables()
        network2 = get_network('VGG_test')
        scope.reuse_variables()
        network3 = get_network('VGG_test')
        scope.reuse_variables()
        network4 = get_network('VGG_test')
        scope.reuse_variables()
        network5 = get_network('VGG_test')
        scope.reuse_variables()
        network6 = get_network('VGG_test')
        scope.reuse_variables()
        network7 = get_network('VGG_test')
        scope.reuse_variables()
        network8 = get_network('VGG_test')
        scope.reuse_variables()
        network9 = get_network('VGG_test')
        scope.reuse_variables()
        network10 = get_network('VGG_test')
        scope.reuse_variables()
        network11 = get_network('VGG_test')
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((200, 200, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)
    
    # front_dress = 'twopropeller_0064'
    # im_names = ['%s_001.png'%(front_dress),'%s_002.png'%(front_dress),'%s_003.png'%(front_dress),'%s_004.png'%(front_dress),'%s_005.png'%(front_dress),'%s_006.png'%(front_dress),'%s_007.png'%(front_dress),'%s_008.png'%(front_dress),'%s_009.png'%(front_dress),'%s_010.png'%(front_dress),'%s_011.png'%(front_dress),'%s_012.png'%(front_dress)]
    im_names = os.listdir('/home/liuxinhai/fine-grained/data/views_chair')
    print(im_names)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name)

