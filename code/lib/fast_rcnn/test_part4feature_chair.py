# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config_chair import cfg
import gt_data_layer.roidb_chair as gdl_roidb
import roi_data_layer.roidb_chair as rdl_roidb
from roi_data_layer.layer_chairv1_test import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
from tensorflow.core.protobuf import saver_pb2
import networks.loupe_attention as lp
import math

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network,network1,network2,network3,network4,network5,network6,network7,network8,network9,network10,network11, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        #with tf.variable_scope("faster_rcnn", reuse=True) as scope:
        self.net = network
        self.net1 = network1
        self.net2 = network2
        self.net3 = network3
        self.net4 = network4
        self.net5 = network5
        self.net6 = network6
        self.net7 = network7
        self.net8 = network8
        self.net9 = network9
        self.net10 = network10
        self.net11 = network11



        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.proposal_number = 20

        self.cluster_size = 512
        self.output_dim = 512
        self.classes = 33
        self.feature_size = 512

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        with tf.variable_scope('build', reuse=True):
            net = self.net

            if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
                # save original values
                with tf.variable_scope('bbox_pred', reuse=True):
                    weights = tf.get_variable("weights")
                    biases = tf.get_variable("biases")

                orig_0 = weights.eval()
                orig_1 = biases.eval()

                # scale and shift with bbox reg unnormalization; then save snapshot
                weights_shape = weights.get_shape().as_list()
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                     if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                        '_iter_{:d}'.format(iter+1) + '.ckpt')
            filename = os.path.join(self.output_dir, filename)

            self.saver.save(sess, filename)
            print 'Wrote snapshot to: {:s}'.format(filename)

            if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
                with tf.variable_scope('bbox_pred', reuse=True):
                    # restore net to original state
                    sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                    sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul
    def faster_rcnn_loss(self, net):
        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(net.get_output('rpn_cls_score_reshape'), [-1, 2])
        rpn_label = tf.reshape(net.get_output('rpn-data')[0], [-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(net.get_output('rpn-data')[1], [0, 2, 3, 1])
        rpn_bbox_inside_weights = tf.transpose(net.get_output('rpn-data')[2], [0, 2, 3, 1])
        rpn_bbox_outside_weights = tf.transpose(net.get_output('rpn-data')[3], [0, 2, 3, 1])
        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                 rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

        # R-CNN
        # classification loss
        cls_score = net.get_output('cls_score')
        label = tf.reshape(net.get_output('roi-data')[1], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = net.get_output('bbox_pred')
        bbox_targets = net.get_output('roi-data')[2]
        bbox_inside_weights = net.get_output('roi-data')[3]
        bbox_outside_weights = net.get_output('roi-data')[4]

        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        return cross_entropy, loss_box , rpn_cross_entropy, rpn_loss_box

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        # blobs = data_layer.forward()
        # netvlad classification loss

        netVLAD = lp.NetVLAD(feature_size=self.feature_size, max_samples=12 * 49 * self.proposal_number,
                             cluster_size=self.cluster_size, output_dim=self.output_dim, gating=False,
                             add_batch_norm=True, is_training=False)
        part_features_fc7 = self.net.get_output('pool_5')
        part_features_fc71 = self.net1.get_output('pool_5')
        part_features_fc72 = self.net2.get_output('pool_5')
        part_features_fc73 = self.net3.get_output('pool_5')
        part_features_fc74 = self.net4.get_output('pool_5')
        part_features_fc75 = self.net5.get_output('pool_5')
        part_features_fc76 = self.net6.get_output('pool_5')
        part_features_fc77 = self.net7.get_output('pool_5')
        part_features_fc78 = self.net8.get_output('pool_5')
        part_features_fc79 = self.net9.get_output('pool_5')
        part_features_fc710 = self.net10.get_output('pool_5')
        part_features_fc711 = self.net11.get_output('pool_5')

        part_features = tf.stack([tf.gather(part_features_fc7, tf.range(self.proposal_number)),
                                  tf.gather(part_features_fc71, tf.range(self.proposal_number))], axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc72, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc73, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc74, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc75, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc76, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc77, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc78, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc79, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc710, tf.range(self.proposal_number))]],
                                  axis=0)
        part_features = tf.concat([part_features, [tf.gather(part_features_fc711, tf.range(self.proposal_number))]],
                                  axis=0)

        # cluster centroids
        vlad_out = netVLAD.forward(tf.reshape(part_features, [12 * 49 * self.proposal_number, self.feature_size]))

        # getting the centroids
        vlad_out = tf.transpose(tf.reshape(vlad_out, [self.feature_size, self.cluster_size]), [1, 0])

        # calculate the two part of the attention
        # first attention part is related to cluster centroids
        square1 = tf.get_variable('square1', [self.cluster_size, self.cluster_size],
                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)))
        part1 = tf.matmul(square1, vlad_out)

        # second attention part is related to the acutual classes
        w_init = tf.truncated_normal_initializer(stddev=0.1)
        b_init = tf.constant_initializer(0.1)
        fc2_w = tf.get_variable('fc2_w', [self.output_dim, self.classes], dtype=tf.float32,
                                initializer=w_init)
        fc2_b = tf.get_variable('fc2_b', [self.classes], dtype=tf.float32, initializer=b_init)
        square2 = tf.get_variable('square2', [self.cluster_size, self.classes],
                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.classes)))
        fc2_wT = tf.transpose(fc2_w, [1, 0])
        part2 = tf.matmul(square2, fc2_wT)

        # calculating the attention
        attention = tf.add(part1, part2)
        w_attention = tf.get_variable('w_attention', [self.output_dim, 1],
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.output_dim)))
        b_attention = tf.get_variable('b_attention', [self.cluster_size, 1],
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)))
        attention = tf.matmul(attention, w_attention) + b_attention
        attention = tf.nn.softmax(attention)
        # add attention to the centroids
        newvlad = tf.multiply(attention, vlad_out)

        attention_combine = tf.reshape(tf.reduce_sum(newvlad, axis=0, keep_dims=True), [-1, self.output_dim])

        # classification layer
        vlad_logits = tf.matmul(attention_combine, fc2_w) + fc2_b
        vlad_prob = tf.nn.softmax(vlad_logits)


        # initializing variables
        saver1 = tf.train.Saver(max_to_keep=150)
        self.saver = saver1
        sess.run(tf.global_variables_initializer())
        self.saver.restore(sess, self.pretrained_model)
        print('loaded:%s'%(self.pretrained_model))



        last_snapshot_iter = -1
        timer = Timer()
        sums = .0

        class_ac_test = True
        # class_ac_test = False
        model_num = 1930
        classes_num = [20,100,100,100,5,10,50,20,5,100,50,100,100,100,
                           40,50,100,50,30,50,100,100,50,100,100,100,20,100,5,5,20,20,30]
        class_acc = np.zeros(self.classes, np.float32)
        # if class_ac_test == True:
        #     model_num = 732
        #     classes_num = [100, 10, 50, 50, 100, 100, 100, 100, 20, 50, 7, 30, 15]
        # else:
        #     model_num = 3991
        #     classes_num = [106, 515, 889, 200, 200, 465, 200, 680, 392, 344]

        for iter in range(model_num):
            # get one batch
            train_target = data_layer.netvlad_target()

            blobs = data_layer.forward()
            blobs1 = data_layer.forward()
            blobs2 = data_layer.forward()
            blobs3 = data_layer.forward()
            blobs4 = data_layer.forward()
            blobs5 = data_layer.forward()
            blobs6 = data_layer.forward()
            blobs7 = data_layer.forward()
            blobs8 = data_layer.forward()
            blobs9 = data_layer.forward()
            blobs10 = data_layer.forward()
            blobs11 = data_layer.forward()

            #raw_input()
            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 1.0,
                       self.net1.data: blobs1['data'], self.net1.im_info: blobs1['im_info'], self.net1.keep_prob: 1.0,
                       self.net2.data: blobs2['data'], self.net2.im_info: blobs2['im_info'], self.net2.keep_prob: 1.0,
                       self.net3.data: blobs3['data'], self.net3.im_info: blobs3['im_info'], self.net3.keep_prob: 1.0,
                       self.net4.data: blobs4['data'], self.net4.im_info: blobs4['im_info'], self.net4.keep_prob: 1.0,
                       self.net5.data: blobs5['data'], self.net5.im_info: blobs5['im_info'], self.net5.keep_prob: 1.0,
                       self.net6.data: blobs6['data'], self.net6.im_info: blobs6['im_info'], self.net6.keep_prob: 1.0,
                       self.net7.data: blobs7['data'], self.net7.im_info: blobs7['im_info'], self.net7.keep_prob: 1.0,
                       self.net8.data: blobs8['data'], self.net8.im_info: blobs8['im_info'], self.net8.keep_prob: 1.0,
                       self.net9.data: blobs9['data'], self.net9.im_info: blobs9['im_info'], self.net9.keep_prob: 1.0,
                       self.net10.data: blobs10['data'], self.net10.im_info: blobs10['im_info'],self.net10.keep_prob: 1.0,
                       self.net11.data: blobs11['data'], self.net11.im_info: blobs11['im_info'],self.net11.keep_prob: 1.0}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()
            test_acc = sess.run(vlad_prob, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            timer.toc()
            if np.argmax(test_acc, axis=1)[0] == np.argmax(train_target):
                sums += 1.0
                class_acc[np.argmax(train_target)] += 1.0

            # print('model id: %d' % iter, np.argmax(test_acc, axis=1)[0], np.argmax(train_target))

        print("Total accuracy: %f" % (sums / model_num))

        fid = open('./chair_bs_4_meanpool.txt', 'a+')
        fid.write('{:.6f}\n'.format(sums / model_num))
        fid.close()

        for i in range(self.classes):
            print("the %d class:%f" % (i, class_acc[i] / classes_num[i]))

        print('class acc: %f'%(sum(class_acc / classes_num) / self.classes))
        fid = open('./chair_bs_class_4_meanpool.txt', 'a+')
        fid.write('{:.6f}\n'.format(sum(class_acc / classes_num) / self.classes))
        fid.close()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if not cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network,network1,network2,network3,network4,network5,network6,network7,network8,network9,network10,network11, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    #os.environ['CUDA_VISIBLE_DEVICES']='0,3'

    roidb = filter_roidb(roidb)
    #saver = tf.train.Saver(max_to_keep=100,write_version=saver_pb2.SaverDef.V1)
    saver = tf.train.Saver(max_to_keep=150)


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, saver, network,network1,network2,network3,network4,network5,network6,network7,network8,network9,network10,network11, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
