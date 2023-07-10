# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import yaml

'''
HED网络类
基于VGG16，所以模型形同VGG16，唯一注意点是concat处，加上concat导致梯度计算不出，不知是某个类库的问题还是其他未知问题
'''

# HED网络定义类
class HED(object):
    def __init__(self, height, width, channel):
        # 定义图片长宽
        self.height = height
        self.width = width
        # 定义占位符
        self.x = tf.placeholder(tf.float32, (None, height, width, channel))
        # 定义配置属性，来自配置文件
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)

    def vgg_hed(self):
        '''
        VGG16模型为2层卷积+reLU,池化，2层卷积+reLU，池化，3层卷积+reLU，池化，3层卷积+reLU，池化，3层卷积+reLU，池化，3层全连接+reLU,softmax输出
        :return: 中间5层图片及最后融合的图片共6个数据
        '''
        # block对应的函数是iteration层卷积+reLU
        bn1, relu1 = self.block(input_tensor=self.x, filters=64, iteration=2, dilation_rate=[(4, 4), (1, 1)], name='block1')
        mp1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool1')

        bn2, relu2 = self.block(input_tensor=mp1, filters=128, iteration=2, name='block2')
        mp2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool2')

        bn3, relu3 = self.block(input_tensor=mp2, filters=256, iteration=3, name='block3')
        mp3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool3')

        bn4, relu4 = self.block(input_tensor=mp3, filters=512, iteration=3, name='block4')
        mp4 = tf.layers.max_pooling2d(inputs=relu4, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool4')

        bn5, relu5 = self.block(input_tensor=mp4, filters=512, iteration=3, name='block5')

        # self.side()对图片进行反卷积
        self.side1 = self.side(input_tensor=bn1, stride=(1, 1), name='side1', deconv=False)
        self.side2 = self.side(input_tensor=bn2, stride=(2, 2), name='side2')
        self.side3 = self.side(input_tensor=bn3, stride=(4, 4), name='side3')
        self.side4 = self.side(input_tensor=bn4, stride=(8, 8), name='side4')
        self.side5 = self.side(input_tensor=bn5, stride=(16, 16), name='side5')
        # sides原本对应side12345的合成，但是优化器迭代concat报错所以先直接采用第五层作为sides
        sides = self.side5
        '''
        t1 = [[1, 2, 3], [4, 5, 6]]  
        t2 = [[7, 8, 9], [10, 11, 12]]  
        tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  
        tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        '''
        # sides = tf.concat(values=[self.side1, self.side2, self.side3, self.side4, self.side5], axis=3)
        # tf.layers.conv2d参数含义：filters输出通道数，kernel_size卷积核大小，strides卷积步长
        # 通过1×1卷积核实现通道数的缩放，sides通过concat后(?,224,224,5)通过卷积核变为(?,224,224,1)
        self.fused_side = tf.layers.conv2d(inputs=sides, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                           use_bias=False, kernel_initializer=tf.constant_initializer(0.2), name='fused_side')
        return self.side1, self.side2, self.side3, self.side4, self.side5, self.fused_side

    def block(self, input_tensor, filters, iteration, dilation_rate=None, name=None):
        '''
        相当于将HED网络分层分块，VGG16模型为2层(卷积+reLU),池化，2层(卷积+reLU)，池化，3层(卷积+reLU)，池化，3层(卷积+reLU)，池化，3层(卷积+reLU)，池化，3层全连接+reLU,softmax输出
        这里一个block对应iteration层的卷积+reLU
        :param input_tensor:输入的tensor
        :param filters:输出通道个数，等价于output_channels
        :param iteration:迭代次数，比如第一个block对应2层的(卷积+reLU)，这里iteration为2
        :param dilation_rate:扩张卷积
        :param name:命名空间，不重要，只在tensorboard可视化时看起来会简美些
        :return:经过(卷积+reLU)处理的tensor
        '''
        # dilation_rate表示扩张卷积，针对的是卷积核的大小，
        # 扩张卷积优点：扩展卷积在保持参数个数不变的情况下增大了卷积核的感受野，同时它可以保证输出的特征映射（feature map）的大小保持不变。
        # dilation_rate默认为（1，1）
        # 扩张卷积应用于图像语义分割问题中下采样会降低图像分辨率、丢失信息的一种卷积思路，所以实际上整个代码只是在VGG16第一层卷积中加入了扩张卷积
        if dilation_rate is None:
            dilation_rate = [(1, 1)]
        if len(dilation_rate) == 1:
            dilation_rate *= iteration

        regularizer = tf.contrib.layers.l2_regularizer(self.cfg['weight_decay_ratio'])
        with tf.variable_scope(name):
            relu = input_tensor
            for it in range(iteration):
                tp_dilation_rate = dilation_rate.pop(0)
                print("hed_net:",tp_dilation_rate)
                conv = tf.layers.conv2d(inputs=relu, filters=filters,
                                        kernel_size=(3, 3), strides=(1, 1), padding='same',
                                        activation=None, use_bias=True,
                                        kernel_regularizer=regularizer,
                                        dilation_rate=tp_dilation_rate,
                                        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.5),
                                        name='conv{:d}'.format(it))
                # bn = tf.layers.batch_normalization(inputs=conv, axis=-1, name='bn{:d}'.format(it))
                bn = conv
                relu = tf.nn.relu(bn, name='relu{:d}'.format(it))
        return relu, relu

    def side(self, input_tensor, stride, name, deconv=True):
        '''
        对图片进行反卷积
        :param input_tensor:输入的tensor
        :param stride:卷积步长，反卷积为扩大倍数
        :param name:命名空间名字
        :param deconv:是否反卷积
        :return:反卷积后的张量
        '''
        with tf.variable_scope(name):
            side = tf.layers.conv2d(inputs=input_tensor, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                    padding='same',
                                    activation=None,
                                    bias_initializer=tf.constant_initializer(value=0),
                                    kernel_initializer=tf.constant_initializer(value=0),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
            if deconv:
                # conv2d_transpose名字虽然叫转置，但实际上就是代表反卷积
                # stride步长，即扩大倍数
                side = tf.layers.conv2d_transpose(inputs=side, filters=1, kernel_size=(2*stride[0], 2*stride[1]),
                                                  strides=stride, padding='same',
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  bias_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.cfg['weight_decay_ratio']),
                                                  activation=None)
            # 以上已经处理反卷积完成，这里在对反卷积完成后的张量做一次双线性插值图片处理
            side = tf.image.resize_images(images=side, size=(self.height, self.width),
                                          method=tf.image.ResizeMethod.BILINEAR)
        return side

    def evaluate(self):
        '''
        评价，暂无，之后可用F1等评价
        :return:
        '''
        # evaluation criteria
        # accuracy

        # precision

        # recall

        # F1 score
        pass

    def summary(self):
        '''
        记录
        :return:
        '''
        max_outputs = 1
        tf.summary.image(name='orig_image_sm', tensor=self.x, max_outputs=max_outputs)
        tf.summary.image(name='side1_im', tensor=tf.sigmoid(self.side1), max_outputs=max_outputs, )
        tf.summary.image(name='side2_im', tensor=tf.sigmoid(self.side2), max_outputs=max_outputs, )
        tf.summary.image(name='side3_im', tensor=tf.sigmoid(self.side3), max_outputs=max_outputs, )
        tf.summary.image(name='side4_im', tensor=tf.sigmoid(self.side4), max_outputs=max_outputs, )
        tf.summary.image(name='side5_im', tensor=tf.sigmoid(self.side5), max_outputs=max_outputs, )
        tf.summary.image(name='fused_side_im', tensor=tf.sigmoid(self.fused_side), max_outputs=max_outputs, )

        tf.summary.histogram(name='side1_hist', values=tf.sigmoid(self.side1))
        tf.summary.histogram(name='side2_hist', values=tf.sigmoid(self.side2))
        tf.summary.histogram(name='side3_hist', values=tf.sigmoid(self.side3))
        tf.summary.histogram(name='side4_hist', values=tf.sigmoid(self.side4))
        tf.summary.histogram(name='side5_hist', values=tf.sigmoid(self.side5))
        tf.summary.histogram(name='fused_side_hist', values=tf.sigmoid(self.fused_side))

    def assign_init_weights(self, sess=None):
        '''
        初始化权重,读取VGG16权重文件进行权重初始化
        :param sess: session
        :return:
        '''
        with open(self.cfg['init_weights'], 'rb') as file:
            weights = np.load(file, encoding='latin1').item()
        with tf.variable_scope('block1', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv1_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv1_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv1_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv1_2'][1]))
        print('assign first block done !')
        with tf.variable_scope('block2', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv2_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv2_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv2_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv2_2'][1]))
        print('assign second block done !')
        with tf.variable_scope('block3', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv3_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv3_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv3_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv3_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv3_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv3_3'][1]))
        print('assign third block done !')
        with tf.variable_scope('block4', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv4_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv4_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv4_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv4_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv4_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv4_3'][1]))
        print('assign fourth block done !')
        with tf.variable_scope('block5', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv5_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv5_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv5_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv5_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv5_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv5_3'][1]))
        weights = None  # gc
        print('assign fifth block done !')
        print('net initializing successfully with vgg16 weights trained by imagenet data')