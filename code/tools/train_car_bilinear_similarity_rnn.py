#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train_car_bilinear_similarity_rnn_attention import get_training_roidb, train_net
from fast_rcnn.config_car import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory_car import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb
import tensorflow as tf
import os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=3, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='%d'%args.device_id
    print('Called with args:')
    print(args)
    sess1 = tf.Session()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name


    with tf.variable_scope("build") as scope:
        network = get_network(args.network_name)
        scope.reuse_variables()
        network1 = get_network(args.network_name)
        scope.reuse_variables()
        network2 = get_network(args.network_name)
        scope.reuse_variables()
        network3 = get_network(args.network_name)
        scope.reuse_variables()
        network4 = get_network(args.network_name)
        scope.reuse_variables()
        network5 = get_network(args.network_name)
        scope.reuse_variables()
        network6 = get_network(args.network_name)
        scope.reuse_variables()
        network7 = get_network(args.network_name)
        scope.reuse_variables()
        network8 = get_network(args.network_name)
        scope.reuse_variables()
        network9 = get_network(args.network_name)
        scope.reuse_variables()
        network10 = get_network(args.network_name)
        scope.reuse_variables()
        network11 = get_network(args.network_name)


    print 'Use network `{:s}` in training'.format(args.network_name)
    #train_net(network, network1, network2, network3, network1, network1, network1, network1, network1, network1,
    #          network1, network1, imdb, roidb, output_dir, \
    #         pretrained_model=args.pretrained_model, \
    #          max_iters=args.max_iters)

    train_net(sess1, network,network1,network2,network3,network4,network5,network6,network7,network8,network9,network10,network11, imdb, roidb, output_dir,\
              pretrained_model=args.pretrained_model,\
             max_iters=args.max_iters)
    # train_net(sess1, network, network4, network8, imdb, roidb, output_dir,\
    #           pretrained_model=args.pretrained_model,\
    #          max_iters=args.max_iters)
