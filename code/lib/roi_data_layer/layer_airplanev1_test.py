# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

from fast_rcnn.config_airplane import cfg
from roi_data_layer.minibatch_airplane import get_minibatch
import numpy as np

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._num_models = len(self._roidb) / 12
        self._shuffle_roidb_inds()

        self._classnum = [100, 10, 50, 50, 100, 100, 100, 100, 20, 50, 7, 30, 15]

        for i in range(len(self._classnum)):
            temp = [1.0 if k == i else 0.0 for k in range(13)]
            temp = [temp for m in range(self._classnum[i])]
            temp = np.array(temp, np.float32)
            if i == 0:
                self._train_target = temp
            else:
                self._train_target = np.vstack((self._train_target, temp))

        self._train_target = np.array(self._train_target)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.arange(self._num_models)
        self._cur = 0
        self._shift = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        
        if cfg.TRAIN.HAS_RPN:
            if self._cur >= self._num_models - 1:
                self._shuffle_roidb_inds()

            db_inds = [self._perm[self._cur]*12+self._shift]
            self._shift = self._shift + 1

            if self._shift % 12 == 0:
                self._cur  = self._cur + 1
                self._shift = 0
        else:
            # sample images
            db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
            i = 0
            while (i < cfg.TRAIN.IMS_PER_BATCH):
                ind = self._perm[self._cur]
                num_objs = self._roidb[ind]['boxes'].shape[0]
                if num_objs != 0:
                    db_inds[i] = ind
                    i += 1

                self._cur += 1
                if self._cur >= len(self._roidb):
                    self._shuffle_roidb_inds()

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()

        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def netvlad_target(self):
        return self._train_target[self._perm[self._cur]]
