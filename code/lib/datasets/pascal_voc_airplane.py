# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb_airplane import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config_airplane import cfg
import pdb


class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__', # always index 0
                         'semantic')
                        #''''air01','air02','air03','air04',
			#'bat01','bat02','bat03','bat04',#'bat05','bat06','bat07','bat08','bat09','bat10','bat11','bat12','bat13','bat14','bat15','bat16',
			#'bed01','bed02','bed03','bed04',#'bed05','bed06','bed07','bed08','bed09','bed10','bed11','bed12','bed13','bed14','bed15','bed16',
			#'ben01','ben02','ben03','ben04',#'ben05','ben06','ben07','ben08','ben09','ben10','ben11','ben12','ben13','ben14','ben15','ben16',
			#'boo01','boo02','boo03','boo04',#'boo05','boo06','boo07','boo08','boo09','boo10','boo11','boo12','boo13','boo14','boo15','boo16',
			#'bot01','bot02','bot03','bot04',#'bot05','bot06','bot07','bot08','bot09','bot10','bot11','bot12','bot13','bot14','bot15','bot16',
			#'bow01','bow02','bow03','bow04',#'bow05','bow06','bow07','bow08','bow09','bow10','bow11','bow12','bow13','bow14','bow15','bow16',
			#'car01','car02','car03','car04',#'car05','car06','car07','car08','car09','car10','car11','car12','car13','car14','car15','car16',
			#'cha01','cha02','cha03','cha04',#'cha05','cha06','cha07','cha08','cha09','cha10','cha11','cha12','cha13','cha14','cha15','cha16',
			#'con01','con02','con03','con04',#'con05','con06','con07','con08','con09','con10','con11','con12','con13','con14','con15','con16',
			#'cup01','cup02','cup03','cup04',#'cup05','cup06','cup07','cup08','cup09','cup10','cup11','cup12','cup13','cup14','cup15','cup16',
			#'cur01','cur02','cur03','cur04',#'cur05','cur06','cur07','cur08','cur09','cur10','cur11','cur12','cur13','cur14','cur15','cur16',
			#'des01','des02','des03','des04',#'des05','des06','des07','des08','des09','des10','des11','des12','des13','des14','des15','des16',
			#'doo01','doo02','doo03','doo04',#'doo05','doo06','doo07','doo08','doo09','doo10','doo11','doo12','doo13','doo14','doo15','doo16',
			#'dre01','dre02','dre03','dre04',#'dre05','dre06','dre07','dre08','dre09','dre10','dre11','dre12','dre13','dre14','dre15','dre16',
			#'flo01','flo02','flo03','flo04',#'flo05','flo06','flo07','flo08','flo09','flo10','flo11','flo12','flo13','flo14','flo15','flo16',
			#'gla01','gla02','gla03','gla04',#'gla05','gla06','gla07','gla08','gla09','gla10','gla11','gla12','gla13','gla14','gla15','gla16',
			#'gui01','gui02','gui03','gui04',#'gui05','gui06','gui07','gui08','gui09','gui10','gui11','gui12','gui13','gui14','gui15','gui16',
			#'key01','key02','key03','key04',#'key05','key06','key07','key08','key09','key10','key11','key12','key13','key14','key15','key16',
			#'lam01','lam02','lam03','lam04',#'lam05','lam06','lam07','lam08','lam09','lam10','lam11','lam12','lam13','lam14','lam15','lam16',
			#'lap01','lap02','lap03','lap04',#'lap05','lap06','lap07','lap08','lap09','lap10','lap11','lap12','lap13','lap14','lap15','lap16',
			##'man01','man02','man03','man04',#'man05','man06','man07','man08','man09','man10','man11','man12','man13','man14','man15','man16',
			#'mon01','mon02','mon03','mon04',#'mon05','mon06','mon07','mon08','mon09','mon10','mon11','mon12','mon13','mon14','mon15','mon16',
			#'nig01','nig02','nig03','nig04',#'nig05','nig06','nig07','nig08','nig09','nig10','nig11','nig12','nig13','nig14','nig15','nig16',
			#'per01','per02','per03','per04',#'per05','per06','per07','per08','per09','per10','per11','per12','per13','per14','per15','per16',
			#'pia01','pia02','pia03','pia04',#'pia05','pia06','pia07','pia08','pia09','pia10','pia11','pia12','pia13','pia14','pia15','pia16',
			#'pla01','pla02','pla03','pla04',#'pla05','pla06','pla07','pla08','pla09','pla10','pla11','pla12','pla13','pla14','pla15','pla16',
			#'rad01','rad02','rad03','rad04',#'rad05','rad06','rad07','rad08','rad09','rad10','rad11','rad12','rad13','rad14','rad15','rad16',
			#'ran01','ran02','ran03','ran04',#'ran05','ran06','ran07','ran08','ran09','ran10','ran11','ran12','ran13','ran14','ran15','ran16',
			#'sin01','sin02','sin03','sin04',#'sin05','sin06','sin07','sin08','sin09','sin10','sin11','sin12','sin13','sin14','sin15','sin16',
			#'sof01','sof02','sof03','sof04',#'sof05','sof06','sof07','sof08','sof09','sof10','sof11','sof12','sof13','sof14','sof15','sof16',
			#'sta01','sta02','sta03','sta04',#'sta05','sta06','sta07','sta08','sta09','sta10','sta11','sta12','sta13','sta14','sta15','sta16',
			#'sto01','sto02','sto03','sto04',#'sto05','sto06','sto07','sto08','sto09','sto10','sto11','sto12','sto13','sto14','sto15','sto16',
			#'tab01','tab02','tab03','tab04',#'tab05','tab06','tab07','tab08','tab09','tab10','tab11','tab12','tab13','tab14','tab15','tab16',
			#'ten01','ten02','ten03','ten04',#'ten05','ten06','ten07','ten08','ten09','ten10','ten11','ten12','ten13','ten14','ten15','ten16',
			#'toi01','toi02','toi03','toi04',#'toi05','toi06','toi07','toi08','toi09','toi10','toi11','toi12','toi13','toi14','toi15','toi16',
			#'tvs01','tvs02','tvs03','tvs04',#'tvs05','tvs06','tvs07','tvs08','tvs09','tvs10','tvs11','tvs12','tvs13','tvs14','tvs15','tvs16',
			#'vas01','vas02','vas03','vas04',#'vas05','vas06','vas07','vas08','vas09','vas10','vas11','vas12','vas13','vas14','vas15','vas16',
			#'war01','war02','war03','war04',#'war05','war06','war07','war08','war09','war10','war11','war12','war13','war14','war15','war16',
			#'xbo01','xbo02','xbo03','xbo04')#'xbo05','xbo06','xbo07','xbo08','xbo09','xbo10','xbo11','xbo12','xbo13','xbo14','xbo15','xbo16')'''

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

        fid = open('/home/liuxinhai/Faster-R-CNN/Faster-RCNN_TF/map_detector_0.5_0.5_rpn_0.5_0.5.txt','a+')
        fid.write('{:.4f}\n'.format(np.mean(aps)))
        fid.close()

        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
