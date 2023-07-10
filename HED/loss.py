# -*- coding: UTF-8 -*-
from __future__ import print_function
import tensorflow as tf
import yaml

'''
损失函数类
1.对应两种损失函数计算方法，calc_loss和focal_loss
2.主要研究了calc_loss,该损失函数涉及两个超参数，is_deep_supervised是否深层监督即是否考虑VGG16中间层的输出图，use_weight_regularizer是否正则化
'''

# 损失函数
class HedLoss(object):
    def __init__(self, sides):
        self.sides = sides  # sides对应hed网络输出的6个图
        self.loss = 0.0     # loss对应calc_loss计算的损失函数
        self.floss = 0.0    # floss对应focal_loss计算的损失函数
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)
        self.label = tf.placeholder(tf.float32, (None, self.cfg['height'], self.cfg['width'], 1))   # 定义标签图占位符
        # self.calc_loss()

    def calc_loss(self):
        '''
        损失函数
        :return: 损失函数计算值
        '''
        # is_deep_supervised是否深层监督，若深层监督则考虑HED网络过程中每一张图的权重并取均值
        # 如果不考虑深层监督，则直接考虑融合后图的损失
        # 个人初步想法：考虑深层监督，则每一次迭代实现分层次的计算损失修改每一层次的权重，再计算融合图，修改权重，实现权重更新粒度更加精确，但是最终效果其实只与融合图有关，所以最后结果尚待测试
        if self.cfg['is_deep_supervised']:
            for n in range(len(self.sides)-1):
                tp_loss = self.cfg['sides_weights'][n] * tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.sides[n], pos_weight=self.cfg['pos_weights'])
                self.loss += tf.reduce_mean(tp_loss)
        self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.sides[-1], pos_weight=self.cfg['pos_weights']))

        # tf.get_collection零存整取获取数据，tf.GraphKeys.REGULARIZATION_LOSSES形同名字，正则化处理
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if self.cfg['use_weight_regularizer']:
            self.loss = tf.add_n(reg_loss) + self.loss

        return self.loss


    def focal_loss(self):
        '''
        另一种损失函数
        :return: 损失函数计算值
        '''
        if self.cfg['is_deep_supervised']:
            for n in range(len(self.sides) - 1):
                sg_p = tf.nn.sigmoid(self.sides[n])
                sg_n = 1.0 - sg_p
                sg_p += 1e-5
                sg_n += 1e-5
                pos_num = tf.reduce_sum(tf.cast(self.label > 0.99, tf.float32))
                neg_num = tf.reduce_sum(tf.cast(self.label < 0.01, tf.float32))

                pos = -self.label*sg_n*sg_n*tf.log(sg_p)
                pos = tf.reduce_sum(pos) / (pos_num+1e-5)

                neg = -(1.0-self.label)*sg_p*sg_p*tf.log(sg_n)
                neg = tf.reduce_sum(neg) / (neg_num+1e-5)
                self.floss = self.floss + 0.25*pos + neg*0.75

        sg_p = tf.nn.sigmoid(self.sides[-1])
        sg_n = 1.0 - sg_p
        sg_p += 1e-5
        sg_n += 1e-5
        pos_num = tf.reduce_sum(tf.cast(self.label > 0.99, tf.float32))
        neg_num = tf.reduce_sum(tf.cast(self.label < 0.01, tf.float32))

        pos = -self.label * sg_n * sg_n * tf.log(sg_p)
        pos = tf.reduce_sum(pos) / (pos_num+1e-5)

        neg = -(1.0 - self.label) * sg_p * sg_p * tf.log(sg_n)
        neg = tf.reduce_sum(neg) / (neg_num+1e-5)
        self.floss = self.floss + 0.25*pos + neg*0.75
        if self.cfg['use_weight_regularizer']:
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.floss += tf.add_n(reg_loss)
        return self.floss


    def summary(self):
        '''
        统计
        :return:
        '''
        tf.summary.scalar(name='loss_sm', tensor=self.loss)
        tf.summary.scalar(name='floss_sm', tensor=self.floss)
        max_outputs = 1
        tf.summary.image(name='label_sm', tensor=self.label, max_outputs=max_outputs, )


