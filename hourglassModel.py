# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 10 19:13:56 2017

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
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import datetime
import os
import scipy.io
from convlstm import ConvGRUCell, ConvLSTMCell

class HourglassModel():
    """ HourglassModel class: (to be renamed)
    Generate TensorFlow model to train and predict Human Pose from images (soon videos)
    Please check README.txt for further information on model management.
    """

    def __init__(self, nFeat=512, nStack=4, nLow=4, outputDim=14, batch_size=16, drop_rate=0.2,lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,logdir_train=None, logdir_test=None, name='hourglass',joints=['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'],cam=[]):
        """ Initializer
        Args:
            nStack				: number of stacks (stage/Hourglass modules)
            nFeat				: number of feature channels on conv layers
            nLow				: number of downsampling (pooling) per module
            outputDim			: number of output Dimension (16 for MPII)
            batch_size			: size of training/testing Batch
            dro_rate			: Rate of neurons disabling for Dropout Layers
            lear_rate			: Learning Rate starting value
            decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
            decay_step			: Step to apply decay
            dataset			: Dataset (class DataGenerator)
            training			: (bool) True for training / False for prediction
            w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
             modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
            name				: name of the model
        """
        self.nStack = nStack
        self.nFeat = nFeat
        self.outDim = outputDim
        self.batchSize = batch_size
        self.training = training
        self.w_summary = w_summary
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.joints = joints
        self.cam = cam

    # ACCESSOR

    # def get_input(self):
    #     """ Returns Input (Placeholder) Tensor
    #     Image Input :
    #         Shape: (None,256,256,3)
    #         Type : tf.float32
    #     Warning:
    #         Be sure to build the model first
    #     """
    #     return self.img

    # def get_output(self):
    #     """ Returns Output Tensor
    #     Output Tensor :
    #         Shape: (None, nbStacks, 64, 64, outputDim)
    #         Type : tf.float32
    #     Warning:
    #         Be sure to build the model first
    #     """
    #     return self.output

    # def get_label(self):
    #     """ Returns Label (Placeholder) Tensor
    #     Image Input :
    #         Shape: (None, nbStacks, 64, 64, outputDim)
    #         Type : tf.float32
    #     Warning:
    #         Be sure to build the model first
    #     """
    #     return self.gtMaps

    # def get_loss(self):
    #     """ Returns Loss Tensor
    #     Image Input :
    #         Shape: (1,)
    #         Type : tf.float32
    #     Warning:
    #         Be sure to build the model first
    #     """
    #     return self.loss

    def get_saver(self):
        """ Returns Saver
        /!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
        Warning:
            Be sure to build the model first
        """
        return self.saver

    def generate_model(self):
        """ Create the complete graph
        """
        startTime = time.time()
        print('CREATE MODEL:')
        # for gpu in self.gpu:
        with tf.device(self.gpu):
            with tf.name_scope('inputs'):
                # Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
                self.img1 = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img1')
                self.img2 = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img2')
                self.img3 = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img3')
                self.img4 = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img4')
                # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
                self.gtMaps1 = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))
                self.gtMaps2 = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))
                self.gtMaps3 = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))
                self.gtMaps4 = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))

                self.gtMaps_3D = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 64, self.outDim))
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')

            self.output1 = self.hourglass(self.img1)
            self.output2 = self.hourglass(self.img2)
            self.output3 = self.hourglass(self.img3)
            self.output4 = self.hourglass(self.img4)

            camera1 = self.cam[0]['camera1'][0]
            camera2 = self.cam[1]['camera2'][0]
            camera3 = self.cam[2]['camera3'][0]
            camera4 = self.cam[3]['camera4'][0]
            self.proj_feat1 = self.proj_splat(self.output1[:,-1,:,:,:], camera1[0:9], camera1[9:12], camera1[12:14],camera1[14:16], camera1[16:19], camera1[19:21])
            self.proj_feat2 = self.proj_splat(self.output2[:,-1,:,:,:], camera2[0:9], camera2[9:12], camera2[12:14],camera2[14:16], camera2[16:19], camera2[19:21])
            self.proj_feat3 = self.proj_splat(self.output3[:,-1,:,:,:], camera3[0:9], camera3[9:12], camera3[12:14],camera3[14:16], camera3[16:19], camera3[19:21])
            self.proj_feat4 = self.proj_splat(self.output4[:,-1,:,:,:], camera4[0:9], camera4[9:12], camera4[12:14],camera4[14:16], camera4[16:19], camera4[19:21])
            # self.proj_feat = tf.stack([self.proj_feat1, self.proj_feat2, self.proj_feat3, self.proj_feat4],axis=1)
            # self.rnn_output, self.rnn_state = self.convgru(self.proj_feat)
            self.proj_feat = tf.concat([self.proj_feat1, self.proj_feat2, self.proj_feat3, self.proj_feat4], axis = 4)
            self.pred_vox = self.grid_unet64(self.proj_feat)
        # with tf.device(self.gpu):
        #     self.pool_grid = self.collapse_dims(self.rnn_state)
        #     self.pred_vox = self.grid_unet64(self.rnn_state)
            # self.pred_vox = self.uncollapse_dims(self.pred_vox,self.batchSize)
            self.prob_vox = tf.nn.sigmoid(self.pred_vox)
            graphTime = time.time()
            print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
            with tf.name_scope('loss'):
                self.loss1 = tf.losses.mean_squared_error(labels=self.gtMaps1, predictions=self.output1)
                self.loss2 = tf.losses.mean_squared_error(labels=self.gtMaps2, predictions=self.output2)
                self.loss3 = tf.losses.mean_squared_error(labels=self.gtMaps3, predictions=self.output3)
                self.loss4 = tf.losses.mean_squared_error(labels=self.gtMaps4, predictions=self.output4)
                self.loss_3D = tf.losses.mean_squared_error(labels=self.gtMaps_3D, predictions=self.prob_vox)
            lossTime = time.time()
            print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')
        with tf.device(self.cpu):
            # with tf.name_scope('accuracy'):
            #     self._accuracy_computation()
            # accurTime = time.time()
            # print('---Acc : Done (' + str(int(abs(accurTime - lossTime))) + ' sec.)')
            with tf.name_scope('steps'):
                self.train_step1 = tf.Variable(0, name='global_step1', trainable=False)
                self.train_step2 = tf.Variable(0, name='global_step2', trainable=False)
                self.train_step3 = tf.Variable(0, name='global_step3', trainable=False)
                self.train_step4 = tf.Variable(0, name='global_step4', trainable=False)
                self.train_step5 = tf.Variable(0, name='global_step5', trainable=False)
            with tf.name_scope('lr'):
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step1, self.decay_step, self.decay,staircase=True, name='learning_rate')
            lrTime = time.time()
            print('---LR : Done (' + str(int(abs(lossTime - lrTime))) + ' sec.)')
        # for gpu in self.gpu:
        with tf.device(self.gpu):
            # with tf.name_scope('rmsprop'):
            #     self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            with tf.name_scope('adam'):
                self.adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            optimTime = time.time()
            print('---Optim : Done (' + str(int(abs(optimTime - lrTime))) + ' sec.)')
            with tf.name_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop1 = self.adam.minimize(self.loss1, self.train_step1)
                    self.train_rmsprop2 = self.adam.minimize(self.loss2, self.train_step2)
                    self.train_rmsprop3 = self.adam.minimize(self.loss3, self.train_step3)
                    self.train_rmsprop4 = self.adam.minimize(self.loss4, self.train_step4)
                    self.train_rmsprop_3D = self.adam.minimize(self.loss_3D, self.train_step5)
            minimTime = time.time()
            print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')
        self.init = tf.global_variables_initializer()
        initTime = time.time()
        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')


        with tf.device(self.cpu):
            with tf.name_scope('training'):
                tf.summary.scalar('loss1', self.loss1, collections=['train'])
                tf.summary.scalar('loss2', self.loss2, collections=['train'])
                tf.summary.scalar('loss3', self.loss3, collections=['train'])
                tf.summary.scalar('loss4', self.loss4, collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
                tf.summary.scalar('loss_3D', self.loss_3D, collections=['train'])
            with tf.name_scope('summary'):
                # add validation test
                tf.summary.scalar('valid_loss1', self.loss1, collections=['test'])
                tf.summary.scalar('valid_loss2', self.loss2, collections=['test'])
                tf.summary.scalar('valid_loss3', self.loss3, collections=['test'])
                tf.summary.scalar('valid_loss4', self.loss4, collections=['test'])
                tf.summary.scalar('valid_loss_3D',self.loss_3D, collections=['test'])

                # for i in range(len(self.joints)):
                #     tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])
        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        # self.weight_op = tf.summary.merge_all('weight')



        endTime = time.time()
        print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')
        del endTime, startTime, initTime, optimTime, minimTime, lrTime, lossTime, graphTime, inputTime

    def restore(self, load=None):
        """ Restore a pretrained model
        Args:
            load	: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            # for gpu in self.gpu:
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary(summary=False)
                if load is not None:
                    print('Loading Trained Model')
                    t = time.time()
                    self.saver.restore(self.Session, load)
                    print('Model Loaded (', time.time() - t, ' sec.)')
                else:
                    print('Please give a Model in args (see README for further information)')

    def _train(self, nEpochs=10, epochSize=1000, saveStep=500, validIter=10):
        """
        """
        print("start training")
        with tf.name_scope('Train'):
            self.generator1 = self.dataset[0]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='train')
            self.valid_gen1 = self.dataset[0]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='valid')
            self.generator2 = self.dataset[1]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='train')
            self.valid_gen2 = self.dataset[1]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='valid')
            self.generator3 = self.dataset[2]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='train')
            self.valid_gen3 = self.dataset[2]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='valid')
            self.generator4 = self.dataset[3]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='train')
            self.valid_gen4 = self.dataset[3]._aux_generator(self.batchSize, self.nStack, normalize=True,sample_set='valid')
            startTime = time.time()
            self.resume = {}
            # self.resume['accur'] = []
            self.resume['loss1'] = []
            self.resume['loss2'] = []
            self.resume['loss3'] = []
            self.resume['loss4'] = []
            # self.resume['err'] = []

            for epoch in range(nEpochs):
                epochstartTime = time.time()
                avg_cost1 = 0.
                cost1 = 0.
                avg_cost2 = 0.
                cost2 = 0.
                avg_cost3 = 0.
                cost3 = 0.
                avg_cost4 = 0.
                cost4 = 0.
                avg_cost3D = 0.
                cost3D = 0.

                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                # Training Set
                for i in range(epochSize):
                    # DISPLAY PROGRESS BAR
                    # TODO : Customize Progress Bar
                    percent = (float(i + 1) / epochSize) * 100
                    num = np.int(20 * percent / 100)
                    tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                    sys.stdout.write('\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[:4] + '%' + ' -cost1: ' + str(cost1)[:8] + ' -avg_loss1: ' + str(avg_cost1)[:8] + ' -cost2: ' + str(cost2)[:8] + ' -avg_loss2: ' + str(avg_cost2)[:8]+ ' -cost3: ' + str(cost3)[:8] + ' -avg_loss3: ' + str(avg_cost3)[:8]+ ' -cost4: ' + str(cost4)[:8] + ' -avg_loss4: ' + str(avg_cost4)[:8]+ ' -avg_loss_3D: ' + str(avg_cost3D)[:8]+' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                    sys.stdout.flush()
                    img_train1, gt_train1, gt_3D, weight_train1 = next(self.generator1)
                    img_train2, gt_train2, weight_train2 = next(self.generator2)
                    img_train3, gt_train3, weight_train3 = next(self.generator3)
                    img_train4, gt_train4, weight_train4 = next(self.generator4)
                    # self.train_order.append(order)
                    if i % saveStep == 0:
                        _, out1, c1, _, out2, c2, _, out3, c3, _, out4, c4, _, vox_out, l3D,summary = self.Session.run([self.train_rmsprop1, self.output1, self.loss1, self.train_rmsprop2, self.output2, self.loss2, self.train_rmsprop3, self.output3, self.loss3, self.train_rmsprop4, self.output4, self.loss4, self.train_rmsprop_3D, self.prob_vox, self.loss_3D, self.train_op],feed_dict={self.img1: img_train1, self.gtMaps1: gt_train1, self.img2: img_train2, self.gtMaps2: gt_train2, self.img3: img_train3, self.gtMaps3: gt_train3, self.img4: img_train4, self.gtMaps4: gt_train4,self.gtMaps_3D:gt_3D})
                        # Save summary (Loss + Accuracy)
                        self.train_summary.add_summary(summary, epoch * epochSize + i)
                        self.train_summary.flush()


                    else:
                        _, out1, c1, _, out2, c2, _, out3, c3, _, out4, c4, _, vox_out, l3D= self.Session.run([self.train_rmsprop1, self.output1, self.loss1, self.train_rmsprop2, self.output2, self.loss2, self.train_rmsprop3, self.output3, self.loss3, self.train_rmsprop4, self.output4, self.loss4, self.train_rmsprop_3D, self.prob_vox, self.loss_3D],feed_dict={self.img1: img_train1, self.gtMaps1: gt_train1, self.img2: img_train2, self.gtMaps2: gt_train2, self.img3: img_train3, self.gtMaps3: gt_train3, self.img4: img_train4, self.gtMaps4: gt_train4,self.gtMaps_3D:gt_3D})
                        # scipy.io.savemat('3D_heatmap.mat',{'heatmap':vox_out})

                    # for i in range(self.batchSize):
                    #     scipy.io.savemat('fmap/' + order[i] + '.mat', {'fmap': out[i, self.nStack - 1]})
                    cost1 += c1
                    avg_cost1 = c1
                    cost2 += c2
                    avg_cost2 = c2
                    cost3 += c3
                    avg_cost3 = c3
                    cost4 += c4
                    avg_cost4 = c4
                    cost3D += l3D
                    avg_cost3D = l3D
                epochfinishTime = time.time()
                # Save Weight (axis = epoch)
                #
                # weight_summary = self.Session.run(self.weight_op, {self.img: img_train, self.gtMaps: gt_train})
                # weight_summary = self.Session.run(self.weight_op, {self.img: img_train, self.gtMaps: gt_train})
                # weight_summary = self.Session.run(self.weight_op, {self.img: img_train, self.gtMaps: gt_train})
                # weight_summary = self.Session.run(self.weight_op, {self.img: img_train, self.gtMaps: gt_train})
                # self.train_summary.add_summary(weight_summary, epoch)
                # self.train_summary.flush()
                # self.weight_summary.add_summary(weight_summary, epoch)
                # self.weight_summary.flush()
                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime - epochstartTime) / epochSize))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session, os.path.join(os.getcwd(), str(self.name + '_' + str(epoch + 1))))
                self.resume['loss1'].append(cost1)
                self.resume['loss2'].append(cost2)
                self.resume['loss3'].append(cost3)
                self.resume['loss4'].append(cost4)

                # Validation Set
                # accuracy_array = np.array([0.0] * len(self.joint_accur))
                # for i in range(validIter):
                #     img_valid, gt_valid, w_valid, _= next(self.generator)
                #     accuracy_pred = self.Session.run(self.joint_accur, feed_dict={self.img: img_valid, self.gtMaps: gt_valid})
                #     accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter
                    # valid_loss += np.array(valid_loss, dtype=np.float32)/validIter
                # print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%')
                # self.resume['accur'].append(accuracy_pred)
                # self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                for i in range(validIter):
                    img_valid_1, gt_valid_1, gt_3D_valid, w_valid1 = next(self.valid_gen1)
                    img_valid_2, gt_valid_2, w_valid2 = next(self.valid_gen2)
                    img_valid_3, gt_valid_3, w_valid3 = next(self.valid_gen3)
                    img_valid_4, gt_valid_4, w_valid4 = next(self.valid_gen4)
                    # self.valid_order.append(order_v)
                    out_v1, valid_loss1, out_v2, valid_loss2, out_v3, valid_loss3, out_v4, valid_loss4, vox_valid, valid_3d, valid_summary = self.Session.run([self.output1, self.loss1,self.output2, self.loss2,self.output3, self.loss3,self.output4, self.loss4, self.prob_vox, self.loss_3D, self.test_op], feed_dict={self.img1: img_valid_1, self.gtMaps1: gt_valid_1,self.img2: img_valid_2, self.gtMaps2: gt_valid_2,self.img3: img_valid_3, self.gtMaps3: gt_valid_3,self.img4: img_valid_4, self.gtMaps4: gt_valid_4,self.gtMaps_3D:gt_3D_valid})
                    # for i in range(self.batchSize):
                    #     scipy.io.savemat('fmap/' + order_v[i] + '.mat', {'fmap': out[i, self.nStack - 1]})
                    valid_loss1 += np.array(valid_loss1, dtype=np.float32) / validIter
                    valid_loss2 += np.array(valid_loss2, dtype=np.float32) / validIter
                    valid_loss3 += np.array(valid_loss3, dtype=np.float32) / validIter
                    valid_loss4 += np.array(valid_loss4, dtype=np.float32) / validIter
                    valid_3d += np.array(valid_3d, dtype=np.float32) / validIter
                    # valid_summary = self.Session.run(self.test_op, feed_dict={self.img1: img_valid_1, self.gtMaps1: gt_valid_1,self.img2: img_valid_2, self.gtMaps2: gt_valid_2,self.img3: img_valid_3, self.gtMaps3: gt_valid_3,self.img4: img_valid_4, self.gtMaps4: gt_valid_4})
                #     valid_summary = self.Session.run(self.test_op, feed_dict={self.img: img_valid, self.gtMaps: gt_valid})
                    self.test_summary.add_summary(valid_summary, epoch*validIter+i)
                    self.test_summary.flush()

                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done, valid_loss1:'+ str(valid_loss1) + ' valid loss2: ' + str(valid_loss2) + ' valid loss3: ' + str(valid_loss3) + ' valid loss4: ' + str(valid_loss4) + ' in ' + str(int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime - epochstartTime) / epochSize))[:4] + ' sec.')
            print('Training Done')
            print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize))
            print('  Final Loss1: ' + str(cost1) + '\n' + '  Relative Loss1: ' + str(100 * self.resume['loss1'][-1] / (self.resume['loss1'][0] + 0.1)) + '%')
            print('  Final Loss2: ' + str(cost2) + '\n' + '  Relative Loss2: ' + str(100 * self.resume['loss2'][-1] / (self.resume['loss2'][0] + 0.1)) + '%')
            print('  Final Loss3: ' + str(cost3) + '\n' + '  Relative Loss3: ' + str(100 * self.resume['loss3'][-1] / (self.resume['loss3'][0] + 0.1)) + '%')
            print('  Final Loss4: ' + str(cost4) + '\n' + '  Relative Loss4: ' + str(100 * self.resume['loss4'][-1] / (self.resume['loss4'][0] + 0.1)) + '%')
            # print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) + '%')
            print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))
            # np.save('train_order',self.train_order)
            # np.save('valid_order',self.valid_order)

    def record_training(self, record):
        """ Record Training Data and Export them in CSV file
        Args:
            record		: record dictionnary
        """
        out_file = open(self.name + '_train_record.csv', 'w')
        for line in range(len(record['accur'])):
            out_string = ''
            labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
            for label in labels:
                out_string += str(label) + ', '
            out_string += '\n'
            out_file.write(out_string)
        out_file.close()
        print('Training Record Saved')

    def training_init(self, nEpochs=10, epochSize=1000, saveStep=500, dataset=None, load=None):
        """ Initialize the training
        Args:
            nEpochs		: Number of Epochs to train
            epochSize		: Size of one Epoch
            saveStep		: Step to save 'train' summary (has to be lower than epochSize)
            dataset		: Data Generator (see generator.py)
            load			: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            # for gpu in self.gpu:
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()
                # with tf.device(self.cpu):
                #     self.saver = tf.train.Saver()
                if load is not None:
                    self.saver.restore(self.Session, load)
                # try:
                #	self.saver.restore(self.Session, load)
                # except Exception:
                #	print('Loading Failed! (Check README file for further information)')
                self._train(nEpochs, epochSize, saveStep, validIter=10)

    def _accuracy_computation(self):
        """ Computes accuracy tensor
        """
        self.joint_accur = []
        for i in range(len(self.joints)):
            self.joint_accur.append(self._accur(self.output[:, self.nStack - 1, :, :, i], self.gtMaps[:, self.nStack - 1, :, :, i],self.batchSize))

    def _define_saver_summary(self, summary=True):
        """ Create Summary and Saver
        Args:
            logdir_train		: Path to train summary directory
            logdir_test		: Path to test summary directory
        """
        if (self.logdir_train == None) or (self.logdir_test == None):
            raise ValueError('Train/Test directory not assigned')
        else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver()
            if summary:
                # for gpu in self.gpu:
                with tf.device(self.gpu):
                    # self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                    # self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.as_graph_def())
                    self.train_summary = tf.summary.FileWriter(self.logdir_train)
                    self.test_summary = tf.summary.FileWriter(self.logdir_test)
                    # self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.Session = tf.Session(config=config)
        #self.Session = tf.Session()
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _init_session(self):
        """ Initialize Session
        """
        print('Session initialization')
        t_start = time.time()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # self.Session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.Session = tf.Session(config=config)
        #self.Session = tf.Session()
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut=128, name='r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128

                r2 = self._residual(pool1, numOut=int(self.nFeat / 2), name='r2')
                r3 = self._residual(r2, numOut=self.nFeat, name='r3')
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack
            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                    drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training,name='dropout')
                    ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, 'VALID', name='conv')
                    ll_[0] = self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
                    out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                    out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, self.nStack - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeat, 'hourglass')
                        drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,name='dropout')
                        ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name='conv')
                        ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                        out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                        out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                if self.nStack > 1:
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,training=self.training, name='dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID','conv')
                        out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')
                return tf.stack(out, axis=1, name='final_output')

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            if self.w_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return conv

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            norm			: Output Tensor
        """
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return norm

    def _conv_block(self, inputs, numOut, name='conv_block'):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """

        with tf.name_scope(name):
            with tf.name_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
                conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
            with tf.name_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
            with tf.name_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,is_training=self.training)
                conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
            return conv_3

    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        """ Skip Layer
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                return conv

    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            return tf.add_n([convb, skipl], name='res_block')

    def _hourglass(self, inputs, n, numOut, name='hourglass'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            return tf.add_n([up_2, up_1], name='out_hg')

    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of max position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x, u_y = self._argmax(u)
        v_x, v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                         tf.to_float(91))

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)



    def proj_splat(self, feats, R, T, f, c, k, p):
        # KRcam = tf.matmul(K, Rcam)
        Rcam = np.array([[R[0],R[1],R[2]],[R[3],R[4],R[5]],[R[6],R[7],R[8]]])
        Tcam = np.array([[T[0]],[T[1]],[T[2]]])

        with tf.variable_scope('ProjSplat'):
            nR, fh, fw, fdim = feats.get_shape().as_list()
            rsz_h = float(fh) / 256
            rsz_w = float(fw) / 256

            # Create voxel grid
            with tf.name_scope('GridCenters'):
                grid_range = tf.range(0,64,dtype=tf.float32)
                self.grid = tf.stack(tf.meshgrid(grid_range, grid_range, grid_range))
                self.rs_grid = tf.reshape(self.grid, [3, -1])
                nV = self.rs_grid.get_shape().as_list()[1]
                # self.rs_grid = tf.concat([self.rs_grid, tf.ones([1, nV])], axis=0)

            # Project grid
            with tf.name_scope('World2Cam'):
                Rcam = tf.cast(Rcam,'float32')
                Tcam = tf.cast(Tcam,'float32')
                temp_p = self.rs_grid - tf.tile(Tcam,[1,nV])
                p_cam = tf.matmul(Rcam, temp_p)
                # p_cam = tf.matmul(Rcam, self.rs_grid)
                x_cam, y_cam, z_cam = p_cam[0,:], p_cam[1,:], p_cam[2,:]
                x_cam = x_cam / z_cam
                y_cam = y_cam / z_cam
                r2 = x_cam**2 + y_cam**2
                # r2 = tf.square(x_cam) + tf.square(y_cam)
                # r4 = tf.square(r2)
                r4 = r2**2
                r6 = tf.pow(r2,3)
                r_temp = tf.stack([r2,r4,r6],axis=0)
                k_temp = np.array([[k[0]],[k[1]],[k[2]]])
                k_temp = tf.tile(k_temp,[1,nV])
                k_temp = tf.cast(k_temp,tf.float32)
                rad_temp = r_temp*k_temp
                rad_temp = tf.reduce_sum(rad_temp,axis=0)
                radial = 1+rad_temp
                tan = p[0]*y_cam + p[1]*x_cam
                x = x_cam*(radial + tan) + p[1]*r2
                y = y_cam*(radial + tan) + p[0]*r2
                # x = x_cam * (1+(k[0]*r2)+(r2**2*k[1])+(r2**3*k[2])) + 2*p[0]*x_cam*y_cam+p[1]*(r2+2*x_cam**2)
                # y = y_cam * (1+(k[0]*r2)+(r2**2*k[1])+(r2**3*k[2])) + 2*p[1]*x_cam*y_cam+p[0]*(r2+2*y_cam**2)
                im_x = f[0]*x + c[0]
                im_y = f[1]*y + c[1]
                # tf.Print(im_x,'im_x')
                # tf.Print(im_y,'im_y')
                
                # im_p = tf.matmul(tf.reshape(KRcam, [-1, 4]), self.rs_grid)
                # im_x, im_y, im_z = im_p[0, :], im_p[1, :], im_p[2, :]
                im_x = im_x * rsz_w
                im_y = im_y * rsz_h
                im_z = (z_cam-743.728)/7424.202
                # self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z
                self.im_x, self.im_y, self.im_z = im_x, im_y, im_z

            # Bilinear interpolation
            with tf.name_scope('BilinearInterp'):
                im_x = tf.clip_by_value(im_x, 0, fw - 1)
                im_y = tf.clip_by_value(im_y, 0, fh - 1)
                im_x0 = tf.cast(tf.floor(im_x), 'int32')
                im_x1 = im_x0 + 1
                im_y0 = tf.cast(tf.floor(im_y), 'int32')
                im_y1 = im_y0 + 1
                im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
                im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

                # ind_grid = tf.range(0, nR)
                # ind_grid = tf.expand_dims(ind_grid, 1)
                # im_ind = tf.tile(ind_grid, [1,nV])

                def _get_gather_inds(x, y):
                    # x = tf.expand_dims(x,0)
                    # x = tf.tile(x,[nR,1])
                    # y = tf.expand_dims(y,0)
                    # y = tf.tile(y,[nR,1])
                    return tf.reshape(tf.stack([y, x], axis=1), [-1, 2])

                # Gather  values
                feats = tf.transpose(feats,[1,2,3,0])
                Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
                Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
                Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
                Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))

                # Calculate bilinear weights
                wa = (im_x1_f - im_x) * (im_y1_f - im_y)
                wb = (im_x1_f - im_x) * (im_y - im_y0_f)
                wc = (im_x - im_x0_f) * (im_y1_f - im_y)
                wd = (im_x - im_x0_f) * (im_y - im_y0_f)
                wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
                wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
                wa = tf.expand_dims(wa,2)
                wb = tf.expand_dims(wb,2)
                wc = tf.expand_dims(wc,2)
                wd = tf.expand_dims(wd,2)
                wa = tf.tile(wa, [1, fdim, self.batchSize])
                wb = tf.tile(wb, [1, fdim, self.batchSize])
                wc = tf.tile(wc, [1, fdim, self.batchSize])
                wd = tf.tile(wd, [1, fdim, self.batchSize])
                self.wa, self.wb, self.wc, self.wd = wa, wb, wc, wd
                self.Ia, self.Ib, self.Ic, self.Id = Ia, Ib, Ic, Id
                Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
                Ibilin = tf.transpose(Ibilin,[2,0,1])
                Ibilin = tf.reshape(Ibilin,[-1,fdim])
            with tf.name_scope('AppendDepth'):
                # Concatenate depth value along ray to feature
                self.im_z = tf.tile(self.im_z,[self.batchSize])
                Ibilin = tf.concat(
                    [Ibilin, tf.reshape(self.im_z, [nV * self.batchSize, 1])], axis=1)
                fdim = Ibilin.get_shape().as_list()[-1]
                self.Ibilin = tf.reshape(Ibilin, [self.batchSize, 64,64,64,fdim])
                self.Ibilin = tf.transpose(self.Ibilin, [0, 2, 1, 3, 4])
        return self.Ibilin


    def convgru(self,grid, kernel=[3, 3, 3], filters=14):
        bs, im_bs, h, w, d, ch = grid.get_shape().as_list()

        conv_gru = ConvGRUCell(
            shape=[h, w, d],
            initializer=slim.initializers.xavier_initializer(),
            kernel=kernel,
            filters=filters)
        seq_length = [im_bs for _ in range(bs)]
        outputs, states = tf.nn.dynamic_rnn(
            conv_gru,
            grid,
            parallel_iterations=64,
            sequence_length=seq_length,
            dtype=tf.float32,
            time_major=False)
        return outputs, states


    def convlstm(self,grid, kernel=[3, 3, 3], filters=14):
        bs, im_bs, h, w, d, ch = grid.get_shape().as_list()

        conv_lstm = ConvLSTMCell(
            shape=[h, w, d],
            initializer=slim.initializers.xavier_initializer(),
            kernel=kernel,
            filters=filters)
        seq_length = [im_bs for _ in range(bs)]
        outputs, states = tf.nn.dynamic_rnn(
            conv_lstm,
            grid,
            parallel_iterations=64,
            sequence_length=seq_length,
            dtype=tf.float32,
            time_major=False)
        return outputs, states


    conv_rnns = {'gru': convgru, 'lstm': convlstm}

    def grid_unet64(self, cost_vol):
        n, h, w, d, ch = cost_vol.get_shape().as_list()
        self.grid_net = {}
        with tf.variable_scope('Grid_Unet'):
            conv1 = self.conv3d('conv1',cost_vol,4,32,activation=None,norm='BN',mode='TRAIN')
            self.grid_net['conv1'] = conv1
            conv2 = self.conv3d('conv2', conv1, 4, 64, norm='BN', mode='TRAIN')
            self.grid_net['conv2'] = conv2
            conv3 = self.conv3d('conv3', conv2, 4, 128, norm='BN', mode='TRAIN')
            self.grid_net['conv3'] = conv3
            conv4 = self.conv3d('conv4', conv3, 4, 256, norm='BN', mode='TRAIN')
            self.grid_net['conv4'] = conv4
            deconv1 = self.deconv3d('deconv1', conv4, 4, 128, norm='BN', mode='TRAIN')
            self.grid_net['deconv1'] = deconv1
            deconv1 = tf.concat([deconv1, conv3], axis=4)
            deconv2 = self.deconv3d('deconv2', deconv1, 4, 64, norm='BN', mode='TRAIN')
            self.grid_net['deconv2'] = deconv2
            deconv2 = tf.concat([deconv2, conv2], axis=4)
            deconv3 = self.deconv3d('deconv3', deconv2, 4, 32, norm='BN', mode='TRAIN')
            self.grid_net['deconv3'] = deconv3
            deconv3 = tf.concat([deconv3, conv1], axis=4)
            deconv4 = self.deconv3d('deconv4', deconv3, 4, 32, norm='BN', mode='TRAIN')
            self.grid_net['deconv4'] = deconv4
            final_vol = self.deconv3d('out', deconv4, 4, 14, stride=1, norm=None, mode='TRAIN')
            self.grid_net['out'] = final_vol

        return final_vol

    def fuse_net(self,vol):
        n,h,w,d,ch = vol.get_shape().as_list()
        with tf.variable_scope("fuse_net"):
            conv1 = self.conv3d('conv1',vol,4,32,activation=None,norm='BN',mode='TRAIN')
            conv2 = self.conv3d('conv2', conv1, 4, 64, norm='BN', mode='TRAIN')
            conv3 = self.conv3d('conv3', conv2, 4, 128, norm='BN', mode='TRAIN')
            conv4 = self.conv3d('conv4', conv3, 4, 256, norm='BN', mode='TRAIN')
            conv5 = self.conv3d('conv5', conv4, 4, 128, norm='BN', mode='TRAIN')
            conv6 = self.conv3d('conv6', conv5, 4, 64, norm='BN', mode='TRAIN')
            conv7 = self.conv3d('conv7', conv6, 4, 32, norm='BN', mode='TRAIN')
            conv8 = self.conv3d('conv8', conv7, 4, 14, stride=1,norm=None, mode='TRAIN')
        return conv8

    def conv3d(self,name,X,fsize,ch,stride=2,norm=None,padding="SAME",activation=tf.nn.relu,mode="TRAIN"):
        bs, h, w, d, in_ch = X.get_shape().as_list()
        filt_shape = [fsize, fsize, fsize, in_ch, ch]
        stride = [1, stride, stride, stride, 1]
        with tf.variable_scope(name):
            if activation is not None:
                X = activation(X)
            params = tf.get_variable(name='weights',shape=filt_shape,initializer=slim.initializers.xavier_initializer())
            # params = get_weights(filt_shape)
            X = tf.nn.conv3d(X, params, stride, padding)
            if norm is None:
                bias_dim = [filt_shape[-1]]
                X = tf.nn.bias_add(X, tf.get_variable(name='bias',shape=bias_dim,initializer=tf.constant_initializer(0.0)))
            elif norm == 'BN':
                is_training = (True if mode == "TRAIN" else False)
                X = slim.batch_norm(
                    X, is_training=is_training, updates_collections=None)
            # elif norm == 'IN':
            #     X = instance_norm(X)
            else:
                print('Invalid normalization! Choose from {None, BN, IN}')

        return X

    def deconv3d(self,name,X,fsize,ch,stride=2,norm=None,padding="SAME",activation=tf.nn.relu,mode="TRAIN"):
        bs, h, w, d, in_ch = X.get_shape().as_list()
        filt_shape = [fsize, fsize, fsize, ch, in_ch]
        out_shape = [bs, h * stride, w * stride, d * stride, ch]
        stride = [1, stride, stride, stride, 1]
        with tf.variable_scope(name):
            if activation is not None:
                X = activation(X)
            params = tf.get_variable(name='weights', shape=filt_shape,initializer=slim.initializers.xavier_initializer())
            # params = get_weights(filt_shape)
            X = tf.nn.conv3d_transpose(X, params, out_shape, stride, padding)
            if norm is None:
                bias_dim = [filt_shape[-2]]
                X = tf.nn.bias_add(X, tf.get_variable(name='bias',shape=bias_dim,initializer=tf.constant_initializer(0.0)))
            elif norm == 'BN':
                is_training = (True if mode == "TRAIN" else False)
                X = slim.batch_norm(
                    X, is_training=is_training, updates_collections=None)
            # elif norm == 'IN':
            #     X = instance_norm(X)
            else:
                print('Invalid normalization! Choose from {None, BN, IN}')

        return X

    def collapse_dims(self,T):
        shape = T.get_shape().as_list()
        return tf.reshape(T, [-1] + shape[2:])

    def uncollapse_dims(self,T, s1, s2):
        shape = T.get_shape().as_list()
        return tf.reshape(T, [s1, s2] + shape[1:])