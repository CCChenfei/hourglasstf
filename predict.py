# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Sep  4 18:11:46 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
	This python code creates a Stacked Hourglass Model
	(Credits : A.Newell et al.)
	(Paper : https://arxiv.org/abs/1603.06937)

	Code translated from 'anewell' github
	Torch7(LUA) --> TensorFlow(PYTHON)
	(Code : https://github.com/anewell/pose-hg-train)

	Modification are made and explained in the report
	Goal : Achieve Real Time detection (Webcam)
	----- Modifications made to obtain faster results (trade off speed/accuracy)

	This work is free of use, please cite the author if you use it!

"""
import sys

sys.path.append('./')

from hourglassModel import HourglassModel
from time import time, clock
import numpy as np
import tensorflow as tf
import scipy.io
from train import process_config
import cv2
from predictClass import PredictProcessor
from datagen import DataGenerator
#import config as cfg


class Inference():
    """ Inference Class
    Use this file to make your prediction
    Easy to Use
    Images used for inference should be RGB images (int values in [0,255])
    Methods:
        webcamSingle : Single Person Pose Estimation on Webcam Stream
        webcamMultiple : Multiple Person Pose Estimation on Webcam Stream
        webcamPCA : Single Person Pose Estimation with reconstruction error (PCA)
        webcamYOLO : Object Detector
        predictHM : Returns Heat Map for an input RGB Image
        predictJoints : Returns joint's location (for a 256x256 image)
        pltSkeleton : Plot skeleton on image
        runVideoFilter : SURPRISE !!!
    """

    def __init__(self, config_file='config.cfg', model='hg_refined_tiny_200'):
        """ Initilize the Predictor
        Args:
            config_file 	 	: *.cfg file with model's parameters
            model 	 	 	 	: *.index file's name. (weights to load)
        """
        t = time()
        params = process_config(config_file)
        datatest1 = DataGenerator(joints_name=params['joint_list'], img_dir_test=params['img_directory_test1'], test_data_file=params['test_txt_file1'],remove_joints=params['remove_joints'])
        datatest1._create_test_table()
        datatest2 = DataGenerator(joints_name=params['joint_list'], img_dir_test=params['img_directory_test2'],test_data_file=params['test_txt_file2'], remove_joints=params['remove_joints'])
        datatest2._create_test_table()
        datatest3 = DataGenerator(joints_name=params['joint_list'], img_dir_test=params['img_directory_test3'],test_data_file=params['test_txt_file3'], remove_joints=params['remove_joints'])
        datatest3._create_test_table()
        datatest4 = DataGenerator(joints_name=params['joint_list'], img_dir_test=params['img_directory_test4'],test_data_file=params['test_txt_file4'], remove_joints=params['remove_joints'])
        datatest4._create_test_table()
        self.predict = PredictProcessor(params)
        self.predict.color_palette()
        self.predict.LINKS_JOINTS()
        self.predict.model_init()
        self.predict.load_model(load=model)
        self.predict._create_prediction_tensor()
        # self.predict.compute_pck(datagen=datatest,idlh=9,idrs=2)
        # self.predict.save_output_as_mat(datagen=datatest,idlh=9,idrs=2)
        self.predict.save_multioutput_as_mat(datagen1=datatest1, datagen2=datatest2,datagen3=datatest3,datagen4=datatest4,idlh=9,idrs=2)
        print('Done: ', time() - t, ' sec.')


    # ----------------------- Heat Map Prediction ------------------------------

    def predictHM(self, img):
        """ Return Sigmoid Prediction Heat Map
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
        """
        return self.predict.pred(self, img / 255, debug=False, sess=None)

    # ------------------------- Joint Prediction -------------------------------

    def predictJoints(self, img, mode='cpu', thresh=0.2):
        """ Return Joint Location
        /!\ Location with respect to 256x256 image
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
            mode : 'cpu' / 'gpu' Select a mode to compute joints' location
            thresh : Joint Threshold
        """
        SIZE = False
        if len(img.shape) == 3:
            batch = np.expand_dims(img, axis=0)
            SIZE = True
        elif len(img.shape) == 4:
            batch = np.copy(img)
            SIZE = True
        if SIZE:
            if mode == 'cpu':
                return self.predict.joints_pred_numpy(self, batch / 255, coord='img', thresh=thresh, sess=None)
            elif mode == 'gpu':
                return self.predict.joints_pred(self, batch / 255, coord='img', debug=False, sess=None)
            else:
                print("Error : Mode should be 'cpu'/'gpu'")
        else:
            print('Error : Input is not a RGB image nor a batch of RGB images')

    # ----------------------------- Plot Skeleton ------------------------------

    def pltSkeleton(self, img, thresh, pltJ, pltL):
        """ Return an image with plotted joints and limbs
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
            thresh: Joint Threshold
            pltJ: (bool) True to plot joints
            pltL: (bool) True to plot limbs
        """
        return self.predict.pltSkeleton(self, img, thresh=thresh, pltJ=pltJ, pltL=pltL, tocopy=True, norm=True)


if __name__ == '__main__':
    Inference('config.cfg','4hg_200')
