# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 17 15:50:43 2017

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
import pdb

sys.path.append('./')

from hourglassModel import HourglassModel
from time import time, clock, sleep
import numpy as np
import tensorflow as tf
import scipy.io
from train import process_config
import cv2
from datagen import DataGenerator


class PredictProcessor():
    """
    PredictProcessor class: Give the tools to open and use a trained model for
    prediction.
    Dependency: OpenCV or PIL (OpenCV prefered)

    Comments:
        Every CAPITAL LETTER methods are to be modified with regard to your needs and dataset
    """

    # -------------------------INITIALIZATION METHODS---------------------------
    def __init__(self, config_dict):
        """ Initializer
        Args:
            config_dict	: config_dict
        """

        self.params = config_dict
        camera1 = scipy.io.loadmat(self.params['camera1'])
        camera2 = scipy.io.loadmat(self.params['camera2'])
        camera3 = scipy.io.loadmat(self.params['camera3'])
        camera4 = scipy.io.loadmat(self.params['camera4'])
        cam = [camera1, camera2, camera3, camera4]
        self.HG = HourglassModel(nFeat=self.params['nfeats'], nStack=self.params['nstacks'], nLow=self.params['nlow'],outputDim=self.params['num_joints'],batch_size=self.params['batch_size'], drop_rate=self.params['dropout_rate'],lear_rate=self.params['learning_rate'],decay=self.params['learning_rate_decay'], decay_step=self.params['decay_step'],dataset=None, training=False,w_summary=True, logdir_test=self.params['log_dir_test'],logdir_train=self.params['log_dir_train'],name=self.params['name'], joints=self.params['joint_list'],cam=cam)
        self.graph = tf.Graph()
        #self.src = 0
        #self.cam_res = (480, 640)

    def color_palette(self):
        """ Creates a color palette dictionnary
        Drawing Purposes
        You don't need to modify this function.
        In case you need other colors, add BGR color code to the color list
        and make sure to give it a name in the color_name list
        /!\ Make sure those 2 lists have the same size
        """
        # BGR COLOR CODE
        self.color = [(241, 242, 224), (196, 203, 128), (136, 150, 0), (64, 77, 0),
                      (201, 230, 200), (132, 199, 129), (71, 160, 67), (32, 94, 27),
                      (130, 224, 255), (7, 193, 255), (0, 160, 255), (0, 111, 255),
                      (220, 216, 207), (174, 164, 144), (139, 125, 96), (100, 90, 69),
                      (252, 229, 179), (247, 195, 79), (229, 155, 3), (155, 87, 1),
                      (231, 190, 225), (200, 104, 186), (176, 39, 156), (162, 31, 123),
                      (210, 205, 255), (115, 115, 229), (80, 83, 239), (40, 40, 198)]
        # Color Names
        self.color_name = ['teal01', 'teal02', 'teal03', 'teal04',
                           'green01', 'green02', 'green03', 'green04',
                           'amber01', 'amber02', 'amber03', 'amber04',
                           'bluegrey01', 'bluegrey02', 'bluegrey03', 'bluegrey04',
                           'lightblue01', 'lightblue02', 'lightblue03', 'lightblue04',
                           'purple01', 'purple02', 'purple03', 'purple04',
                           'red01', 'red02', 'red03', 'red04']
        self.color_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.palette = {}
        for i, name in enumerate(self.color_name):
            self.palette[name] = self.color[i]

    def LINKS_JOINTS(self):
        """ Defines links to be joined for visualization
        Drawing Purposes
        You may need to modify this function
        """
        self.links = {}
        # Edit Links with your needed skeleton
        LINKS = [(0, 1), (1, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7), (2, 8), (8, 10), (10, 12), (3, 9), (9, 11), (11, 13), (8, 9)]
        #self.LINKS_ACP = [(0, 1), (1, 2), (3, 4), (4, 5), (7, 8), (8, 9), (10, 11), (11, 12)]
        color_id = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 20, 21, 22, 25]
        #self.color_id_acp = [8, 9, 9, 8, 19, 20, 20, 19]
        # 13 joints version
        # LINKS = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(6,11),(11,12),(12,13),(6,11),(10,9),(9,8)]
        # color_id = [1,2,3,2,1,0,27,26,25,27,26,25]
        # 10 lines version
        # LINKS = [(0,1),(1,2),(3,4),(4,5),(6,8),(8,9),(13,14),(14,15),(12,11),(11,10)]
        for i in range(len(LINKS)):
            self.links[i] = {'link': LINKS[i], 'color': self.palette[self.color_name[color_id[i]]]}

    # ----------------------------TOOLS----------------------------------------
    def col2RGB(self, col):
        """
        Args:
            col 	: (int-tuple) Color code in BGR MODE
        Returns
            out 	: (int-tuple) Color code in RGB MODE
        """
        return col[::-1]

    def givePixel(self, link, joints):
        """ Returns the pixel corresponding to a link
        Args:
            link 	: (int-tuple) Tuple of linked joints
            joints	: (array) Array of joints position shape = outDim x 2
        Returns:
            out		: (tuple) Tuple of joints position
        """
        return (joints[link[0]].astype(np.int), joints[link[1]].astype(np.int))

    # ---------------------------MODEL METHODS---------------------------------
    def model_init(self):
        """ Initialize the Hourglass Model
        """
        t = time()
        with self.graph.as_default():
            self.HG.generate_model()
        print('Graph Generated in ', int(time() - t), ' sec.')

    def load_model(self, load=None):
        """ Load pretrained weights (See README)
        Args:
            load : File to load
        """
        with self.graph.as_default():
            self.HG.restore(load)


    def _create_joint_tensor(self, tensor, name='joint_tensor', debug=False):
        """ TensorFlow Computation of Joint Position
        Args:
            tensor		: Prediction Tensor Shape [nbStack x 64 x 64 x outDim] or [64 x 64 x outDim]
            name		: name of the tensor
        Returns:
            out			: Tensor of joints position

        Comment:
            Genuinely Agreeing this tensor is UGLY. If you don't trust me, look at
            'prediction' node in TensorBoard.
            In my defence, I implement it to compare computation times with numpy.
        """
        with tf.name_scope(name):
            shape = tensor.get_shape().as_list()
            if debug:
                print(shape)
            if len(shape) == 3:
                resh = tf.reshape(tensor[:, :, 0], [-1])
            elif len(shape) == 4:
                resh = tf.reshape(tensor[-1, :, :, 0], [-1])
            if debug:
                print(resh)
            arg = tf.argmax(resh, 0)
            if debug:
                print(arg, arg.get_shape(), arg.get_shape().as_list())
            joints = tf.expand_dims(tf.stack([arg // tf.to_int64(shape[1]), arg % tf.to_int64(shape[1])], axis=-1),axis=0)
            for i in range(1, shape[-1]):
                if len(shape) == 3:
                    resh = tf.reshape(tensor[:, :, i], [-1])
                elif len(shape) == 4:
                    resh = tf.reshape(tensor[-1, :, :, i], [-1])
                arg = tf.argmax(resh, 0)
                j = tf.expand_dims(tf.stack([arg // tf.to_int64(shape[1]), arg % tf.to_int64(shape[1])], axis=-1),axis=0)
                joints = tf.concat([joints, j], axis=0)
            return tf.identity(joints, name='joints')


    def _create_prediction_tensor(self):
        """ Create Tensor for prediction purposes
        """
        with self.graph.as_default():
            with tf.name_scope('prediction'):
                self.HG.pred_sigmoid1 = tf.nn.sigmoid(self.HG.output1[:, self.HG.nStack - 1],name='sigmoid_final_prediction')
                self.HG.pred_final1 = self.HG.output1[:,self.HG.nStack - 1]
                # self.HG.joint_tensor1 = self._create_joint_tensor1(self.HG.output1[0], name='joint_tensor')
                # self.HG.joint_tensor_final1 = self._create_joint_tensor1(self.HG.output1[0, -1], name='joint_tensor_final')
                self.HG.pred_sigmoid2 = tf.nn.sigmoid(self.HG.output2[:, self.HG.nStack - 1],name='sigmoid_final_prediction')
                self.HG.pred_final2 = self.HG.output2[:, self.HG.nStack - 1]
                # self.HG.joint_tensor = self._create_joint_tensor(self.HG.output1[0], name='joint_tensor')
                # self.HG.joint_tensor_final = self._create_joint_tensor(self.HG.output1[0, -1],name='joint_tensor_final')
                self.HG.pred_sigmoid3 = tf.nn.sigmoid(self.HG.output3[:, self.HG.nStack - 1],name='sigmoid_final_prediction')
                self.HG.pred_final3 = self.HG.output3[:, self.HG.nStack - 1]
                # self.HG.joint_tensor = self._create_joint_tensor(self.HG.output1[0], name='joint_tensor')
                # self.HG.joint_tensor_final = self._create_joint_tensor(self.HG.output1[0, -1],name='joint_tensor_final')
                self.HG.pred_sigmoid4 = tf.nn.sigmoid(self.HG.output4[:, self.HG.nStack - 1],name='sigmoid_final_prediction')
                self.HG.pred_final4 = self.HG.output4[:, self.HG.nStack - 1]
                # self.HG.joint_tensor = self._create_joint_tensor(self.HG.output1[0], name='joint_tensor')
                # self.HG.joint_tensor_final = self._create_joint_tensor(self.HG.output1[0, -1],name='joint_tensor_final')
                self.HG.pred3D = self.HG.prob_vox
        print('Prediction Tensors Ready!')

    # ----------------------------PREDICTION METHODS----------------------------
    def predict_coarse(self, img, debug=False, sess=None):
        ''' Given a 256 x 256 image, Returns prediction Tensor
        This prediction method returns a non processed Output
        Values not Bounded
        Args:
            img		: Image -Shape (256 x256 x 3) -Type : float32
            debug	: (bool) True to output prediction time
        Returns:
            out		: Array -Shape (nbStacks x 64 x 64 x outputDim) -Type : float32
        '''
        if debug:
            t = time()
        if img.shape == (256, 256, 3):
            if sess is None:
                out = self.HG.Session.run(self.HG.output, feed_dict={self.HG.img: np.expand_dims(img, axis=0)})
            else:
                out = sess.run(self.HG.output, feed_dict={self.HG.img: np.expand_dims(img, axis=0)})
        else:
            print('Image Size does not match placeholder shape')
            raise Exception
        if debug:
            print('Pred: ', time() - t, ' sec.')
        return out

    def pred(self, img, debug=False, sess=None):
        """ Given a 256 x 256 image, Returns prediction Tensor
        This prediction method returns values in [0,1]
        Use this method for inference
        Args:
            img		: Image -Shape (256 x256 x 3) -Type : float32
            debug	: (bool) True to output prediction time
        Returns:
            out		: Array -Shape (64 x 64 x outputDim) -Type : float32
        """
        if debug:
            t = time()
        if img.shape == (256, 256, 3):
            img = img.astype(np.float32)/255
            if sess is None:
                out = self.HG.Session.run(self.HG.pred_final, feed_dict={self.HG.img: np.expand_dims(img, axis=0)})
            else:
                out = sess.run(self.HG.pred_final, feed_dict={self.HG.img: np.expand_dims(img, axis=0)})
        else:
            print('Image Size does not match placeholder shape')
            raise Exception
        if debug:
            print('Pred: ', time() - t, ' sec.')
        return out

    def pred_multiview(self, img1, img2, img3, img4, debug=False, sess=None):
        """ Given four 256 x 256 image, Returns prediction Tensor
        This prediction method returns values in [0,1]
        Use this method for inference
        Args:
            img		: Image -Shape (256 x256 x 3) -Type : float32
            debug	: (bool) True to output prediction time
        Returns:
            out		: Array -Shape (64 x 64 x outputDim) -Type : float32
        """
        if debug:
            t = time()
        if img1.shape == (256, 256, 3):
            img1 = img1.astype(np.float32) / 255
        else:
            print('Image1 Size does not match placeholder shape')
            raise Exception
        if img2.shape == (256, 256, 3):
            img2 = img2.astype(np.float32) / 255
        else:
            print('Image2 Size does not match placeholder shape')
            raise Exception
        if img3.shape == (256, 256, 3):
            img3 = img3.astype(np.float32) / 255
        else:
            print('Image3 Size does not match placeholder shape')
            raise Exception
        if img4.shape == (256, 256, 3):
            img4 = img4.astype(np.float32) / 255
        else:
            print('Image4 Size does not match placeholder shape')
            raise Exception
        if sess is None:
            # out1,out2,out3,out4 = self.HG.Session.run(self.HG.pred_final1, self.HG.pred_final2, self.HG.pred_final3, self.HG.pred_final4, feed_dict={self.HG.img1: np.expand_dims(img1, axis=0),self.HG.img2: np.expand_dims(img2, axis=0),self.HG.img3: np.expand_dims(img3, axis=0),self.HG.img4: np.expand_dims(img4, axis=0)})
            # out1 = self.HG.Session.run(self.HG.pred_final1, feed_dict={self.HG.img1: np.expand_dims(img1, axis=0)})
            # out2 = self.HG.Session.run(self.HG.pred_final2, feed_dict={self.HG.img2: np.expand_dims(img2, axis=0)})
            # out3 = self.HG.Session.run(self.HG.pred_final3, feed_dict={self.HG.img3: np.expand_dims(img3, axis=0)})
            # out4 = self.HG.Session.run(self.HG.pred_final4, feed_dict={self.HG.img4: np.expand_dims(img4, axis=0)})
            out3D, out1, out2, out3, out4 = self.HG.Session.run([self.HG.pred_3D,self.HG.pred_final1,self.HG.pred_final2,self.HG.pred_final3,self.HG.pred_final4], feed_dict={self.HG.img1:np.expand_dims(img1,axis=0),self.HG.img2: np.expand_dims(img2, axis=0),self.HG.img3: np.expand_dims(img3, axis=0),self.HG.img4: np.expand_dims(img4, axis=0)})
        else:
            # out1,out2,out3,out4 = sess.run(self.HG.pred_final1, self.HG.pred_final2, self.HG.pred_final3, self.HG.pred_final4, feed_dict={self.HG.img1: np.expand_dims(img1, axis=0),self.HG.img2: np.expand_dims(img2, axis=0),self.HG.img3: np.expand_dims(img3, axis=0),self.HG.img4: np.expand_dims(img4, axis=0)})
            # out1 = sess.run(self.HG.pred_final1, feed_dict={self.HG.img1: np.expand_dims(img1, axis=0)})
            # out2 = sess.run(self.HG.pred_final2, feed_dict={self.HG.img2: np.expand_dims(img2, axis=0)})
            # out3 = sess.run(self.HG.pred_final3, feed_dict={self.HG.img3: np.expand_dims(img3, axis=0)})
            # out4 = sess.run(self.HG.pred_final4, feed_dict={self.HG.img4: np.expand_dims(img4, axis=0)})
            out3D, out1, out2, out3, out4 = sess.run([self.HG.pred_3D, self.HG.pred_final1, self.HG.pred_final2, self.HG.pred_final3, self.HG.pred_final4],feed_dict={self.HG.img1: np.expand_dims(img1, axis=0), self.HG.img2: np.expand_dims(img2, axis=0),self.HG.img3: np.expand_dims(img3, axis=0), self.HG.img4: np.expand_dims(img4, axis=0)})
        if debug:
            print('Pred: ', time() - t, ' sec.')
        out = [out1,out2,out3,out4]
        return out,out3D

    def joints_pred(self, img, coord='hm', debug=False, sess=None):
        """ Given an Image, Returns an array with joints position
        Args:
            img		: Image -Shape (256 x 256 x 3) -Type : float32
            coord	: 'hm'/'img' Give pixel coordinates relative to heatMap('hm') or Image('img')
            debug	: (bool) True to output prediction time
        Returns
            out		: Array -Shape(num_joints x 2) -Type : int
        """
        if debug:
            t = time()
            if sess is None:
                j1 = self.HG.Session.run(self.HG.joint_tensor, feed_dict={self.HG.img: img})
            else:
                j1 = sess.run(self.HG.joint_tensor, feed_dict={self.HG.img: img})
            print('JT:', time() - t)
            t = time()
            if sess is None:
                j2 = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict={self.HG.img: img})
            else:
                j2 = sess.run(self.HG.joint_tensor_final, feed_dict={self.HG.img: img})
            print('JTF:', time() - t)
            if coord == 'hm':
                return j1, j2
            elif coord == 'img':
                return j1 * self.params['img_size'] / self.params['hm_size'], j2 * self.params['img_size'] / \
                       self.params['hm_size']
            else:
                print("Error: 'coord' argument different of ['hm','img']")
        else:
            if sess is None:
                j = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict={self.HG.img: img})
            else:
                j = sess.run(self.HG.joint_tensor_final, feed_dict={self.HG.img: img})
            if coord == 'hm':
                return j
            elif coord == 'img':
                return j * self.params['img_size'] / self.params['hm_size']
            else:
                print("Error: 'coord' argument different of ['hm','img']")

    def joints_pred_numpy(self, img, coord='hm', thresh=0.2, sess=None):
        """ Create Tensor for joint position prediction
        NON TRAINABLE
        TO CALL AFTER GENERATING GRAPH
        Notes:
            Not more efficient than Numpy, prefer Numpy for such operation!
        """
        if sess is None:
            hm = self.HG.Session.run(self.HG.pred_final, feed_dict={self.HG.img: img})
        else:
            hm = sess.run(self.HG.pred_final, feed_dict={self.HG.img: img})
        joints = -1 * np.ones(shape=(self.params['num_joints'], 2))
        for i in range(self.params['num_joints']):
            index = np.unravel_index(hm[0, :, :, i].argmax(), (self.params['hm_size'], self.params['hm_size']))
            if hm[0, index[0], index[1], i] > thresh:
                if coord == 'hm':
                    joints[i] = np.array(index)
                elif coord == 'img':
                    joints[i] = np.array(index) * self.params['img_size'] / self.params['hm_size']
        return joints

    def batch_pred(self, batch, debug=False):
        """ Given a 256 x 256 images, Returns prediction Tensor
        This prediction method returns values in [0,1]
        Use this method for inference
        Args:
            batch	: Batch -Shape (batchSize x 256 x 256 x 3) -Type : float32
            debug	: (bool) True to output prediction time
        Returns:
            out		: Array -Shape (batchSize x 64 x 64 x outputDim) -Type : float32
        """
        if debug:
            t = time()
        if batch[0].shape == (256, 256, 3):
            out = self.HG.Session.run(self.HG.pred_final, feed_dict={self.HG.img: batch})
        else:
            print('Image Size does not match placeholder shape')
            raise Exception
        if debug:
            print('Pred: ', time() - t, ' sec.')
        return out

    # -------------------------------PLOT FUNCTION------------------------------
    def plt_skeleton(self, img, tocopy=True, debug=False, sess=None):
        """ Given an Image, returns Image with plotted limbs (TF VERSION)
        Args:
            img 	: Source Image shape = (256,256,3)
            tocopy 	: (bool) False to write on source image / True to return a new array
            debug	: (bool) for testing puposes
            sess	: KEEP NONE
        """
        joints = self.joints_pred(np.expand_dims(img, axis=0), coord='img', debug=False, sess=sess)
        if tocopy:
            img = np.copy(img)
        for i in range(len(self.links)):
            position = self.givePixel(self.links[i]['link'], joints)
            cv2.line(img, tuple(position[0])[::-1], tuple(position[1])[::-1], self.links[i]['color'][::-1], thickness=2)
        if tocopy:
            return img

    def plt_skeleton_numpy(self, img, tocopy=True, thresh=0.2, sess=None, joint_plt=True):
        """ Given an Image, returns Image with plotted limbs (NUMPY VERSION)
        Args:
            img			: Source Image shape = (256,256,3)
            tocopy		: (bool) False to write on source image / True to return a new array
            thresh		: Joint Threshold
            sess		: KEEP NONE
            joint_plt	: (bool) True to plot joints (as circles)
        """
        joints = self.joints_pred_numpy(np.expand_dims(img, axis=0), coord='img', thresh=thresh, sess=sess)
        if tocopy:
            img = np.copy(img) * 255
        for i in range(len(self.links)):
            l = self.links[i]['link']
            good_link = True
            for p in l:
                if np.array_equal(joints[p], [-1, -1]):
                    good_link = False
            if good_link:
                position = self.givePixel(self.links[i]['link'], joints)
                cv2.line(img, tuple(position[0])[::-1], tuple(position[1])[::-1], self.links[i]['color'][::-1],
                         thickness=2)
        if joint_plt:
            for p in range(len(joints)):
                if not (np.array_equal(joints[p], [-1, -1])):
                    cv2.circle(img, (int(joints[p, 1]), int(joints[p, 0])), radius=3, color=self.color[p][::-1],
                               thickness=-1)
        if tocopy:
            return img

    def pltSkeleton(self, img, thresh=0.2, pltJ=True, pltL=True, tocopy=True, norm=True):
        """ Plot skeleton on Image (Single Detection)
        Args:
            img			: Input Image ( RGB IMAGE 256x256x3)
            thresh		: Joints Threshold
            pltL		: (bool) Plot Limbs
            tocopy		: (bool) Plot on imput image or return a copy
            norm		: (bool) Normalize input Image (DON'T MODIFY)
        Returns:
            img			: Copy of input image if 'tocopy'
        """
        # img_hg = np.zeros((64,64,3),np.float32)
        if tocopy:
            img = np.copy(img)
        if norm:
            img_hg = img / 255
            hg = self.HG.Session.run(self.HG.pred_final, feed_dict={self.HG.img: np.expand_dims(img_hg, axis=0)})
        else:
            hg = self.HG.Session.run(self.HG.pred_final, feed_dict={self.HG.img: np.expand_dims(img, axis=0)})
        j = np.ones(shape=(self.params['num_joints'], 2)) * -1
        for i in range(len(j)):
            idx = np.unravel_index(hg[0, :, :, i].argmax(), (64, 64))
            if hg[0, idx[0], idx[1], i] > thresh:
                j[i] = np.asarray(idx) * 256 / 64
                if pltJ:
                    cv2.circle(img, center=tuple(j[i].astype(np.int))[::-1], radius=5, color=self.color[i][::-1],
                               thickness=-1)
        if pltL:
            for i in range(len(self.links)):
                l = self.links[i]['link']
                good_link = True
                for p in l:
                    if np.array_equal(j[p], [-1, -1]):
                        good_link = False
                if good_link:
                    pos = self.givePixel(l, j)
                    cv2.line(img, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness=5)
        if tocopy:
            return img

        # -------------------------Benchmark Methods (PCK)-------------------------

    def pcki(self, joint_id, gtJ, prJ, idlh=9, idrs=2):
        """ Compute PCK accuracy on a given joint
        Args:
            joint_id	: Index of the joint considered
            gtJ			: Ground Truth Joint
            prJ			: Predicted Joint
            idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
            idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
        Returns:
            (float) NORMALIZED L2 ERROR
        """
        return np.linalg.norm(gtJ[joint_id] - prJ[joint_id][::-1]) / np.linalg.norm(gtJ[idlh] - gtJ[idrs])

    def pck(self, weight, gtJ, prJ, idlh=9, idrs=2):
        """ Compute PCK accuracy for a sample
        Args:
            weight		: Index of the joint considered
            gtJFull	: Ground Truth (sampled on whole image)
            gtJ			: Ground Truth (sampled on reduced image)
            prJ			: Prediction
            boxL		: Box Lenght
            idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
            idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
        """
        for i in range(len(weight)):
            if weight[i] == 1:
                self.ratio_pck.append(self.pcki(i, gtJ, prJ, idlh=idlh, idrs=idrs))
                #self.ratio_pck_full.append(self.pcki(i, gtJFull, np.asarray(prJ / 255 * boxL)))
                self.pck_id.append(i)

    def compute_pck(self, datagen, idlh=9, idrs=2, testSet=None):
        """ Compute PCK on dataset
        Args:
            datagen	: (DataGenerator)
            idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
            idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
        """
        datagen.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        self.ratio_pck = []
        #self.ratio_pck_full = []
        self.pck_id = []
        self.pck_mean = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
       # np.array(self.pck_mean)
        samples = len(datagen.pck_samples)
        startT = time()
        for idx, sample in enumerate(datagen.pck_samples):
            percent = (float(idx + 1) / samples) * 100
            num = np.int(20 * percent / 100)
            tToEpoch = int((time() - startT) * (100 - percent) / (percent))
            sys.stdout.write('\r PCK : {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                          :4] + '%' + ' -timeToEnd: ' + str(
                tToEpoch) + ' sec.')
            sys.stdout.flush()
            res = datagen.getSample(sample)
            if res != False:
                img, gtJoints, w = res

                prJoints = self.joints_pred_numpy(np.expand_dims(img / 255, axis=0), coord='hm', thresh=0)
                self.pck(w, gtJoints, prJoints, idlh=idlh, idrs=idrs)
                self.pck_mean = [i + j for i,j in zip(self.pck_mean,self.ratio_pck)]
                #self.pck_mean = np.array(self.ratio_pck) + np.array(self.pck_mean)
                img_to_show = self.pltSkeleton(img,thresh=0.2,pltJ=True,pltL=True,tocopy=True,norm=False)
                cv2.imshow("img",img_to_show)
                cv2.waitKey()
                # while(cv2.waitKey()==27):
                #     pass
        #self.pck_mean = np.array(self.pck_mean) / len(datagen.pck_samples)
        self.pck_mean = [k / samples for k in self.pck_mean]
        #print('pck：','head ',self.pck_mean[0], 'neck ',self.pck_mean[1], 'r_shoulder ',self.pck_mean[2], 'l_shoulder ', self.pck_mean[3],'r_elbow ',self.pck_mean[4], 'l_elbow ',self.pck_mean[5], 'r_wrist ',self.pck_mean[6], 'l_wrist ',self.pck_mean[7], 'r_hip ',self.pck_mean[8], 'l_hip ',self.pck_mean[9], 'r_knee ',self.pck_mean[10], 'l_knee ',self.pck_mean[11], 'r_anckle ',self.pck_mean[12], 'l_anckle ',self.pck_mean[13])
        print('Done in ', int(time() - startT), 'sec.')

    def save_output_as_mat(self, datagen, idlh=9, idrs=2, testSet=None):
        datagen.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        samples = len(datagen.pck_samples)
        # out = [None]*samples
        # with open('testoutput.mat', 'a')as f:
        for idx, sample in enumerate(datagen.pck_samples):
            res = datagen.getSample(sample)
            if res != False:
                img, gtJoints, w = res
                out1 = self.pred(img)
                out = out1[0]
                # with open('testoutput.mat', 'a')as f:
                #     scipy.io.savemat(f, {'joint'+str(idx+1): out1})
                scipy.io.savemat('joints_human3.6M/joint'+str(idx+1)+'.mat',{'joint':out})
                # if(idx>20):
                #     break
        # f.close()

        # scipy.io.savemat('testoutput.mat', {'out':out})
        print('Done for save')
    # def getJointsDist(self, gtJ, prJ):
    # def compute_pcp(self, datagen,):
    def save_multioutput_as_mat(self, datagen1, datagen2, datagen3, datagen4, idlh=9, idrs=2, testSet=None):
        datagen1.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        datagen2.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        datagen3.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        datagen4.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        samples = len(datagen1.pck_samples)
        for idx in range(samples):
            sample1 = datagen1.pck_samples[idx]
            sample2 = datagen2.pck_samples[idx]
            sample3 = datagen3.pck_samples[idx]
            sample4 = datagen4.pck_samples[idx]
            res1 = datagen1.getSample(sample1)
            res2 = datagen2.getSample(sample2)
            res3 = datagen3.getSample(sample3)
            res4 = datagen4.getSample(sample4)
            img1, gtJoints1, w1 = res1
            img2, gtJoints2, w2 = res2
            img3, gtJoints3, w3 = res3
            img4, gtJoints4, w4 = res4
            out, out3D= self.pred_multiview(img1,img2,img3,img4)
            scipy.io.savemat('joints_multiview/joint'+str(idx+1)+'.mat',{'joint':out})
            scipy.io.savemat('joints_multiview/joint3D'+str(idx+1)+'.mat',{'joint3D':out3D})

# if __name__ == '__main__':
#     t = time()
#     params = process_config('configTiny.cfg')
#     predict = PredictProcessor(params)
#     predict.color_palette()
#     predict.LINKS_JOINTS()
#     predict.model_init()
#     predict.load_model(load='hg_refined_tiny_200')
#     predict._create_prediction_tensor()
#     print('Done: ', time() - t, ' sec.')