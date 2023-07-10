# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import yaml
from hed_net import HED
from loss import HedLoss
import os
import cv2
import argparse
import random
from time import time
import matplotlib.pyplot as plt

'''
训练HED网络主函数
1.超参数通过cfg.yml配置更改
2.采用batch的模式训练，每一轮迭代遍历所有训练集(处理时会打乱顺序）,本机对于224×224图片来说，batch_size=2能够不卡顿的运行，可以适当增加batch_size提高迭代速度
3.暂未写损失函数可视化！
'''

# dataSet存放数据文件，包括图片信息和标签信息
class DataSet(object):
    def __init__(self):
        # 读取配置文件
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)
        self.imgs = None    # 图片信息
        self.labels = None  # 标签信息
        self.samples_num = 0    # 样本数量
        self.read_data()    # 调用函数加载图片和标签

    def read_data(self):
        '''
        读取训练图片文件和标签图片文件
        :return:
        '''
        img_names = []
        label_names = []
        # 配置文件cfg.yml中file_name对应train.txt，记录一一对应的训练图片与标签图片，存放在img_names,label_names,为之后读取做准备
        with open(self.cfg['file_name']) as file:
            while True:
                il = file.readline(1500)    # 如果样本数据大于1500，修改该值
                if not il:
                    break
                a = il.split(sep=' ')
                img_names.append(a[0])
                label_names.append(a[1][0:-1])  # remove '\n'
        self.samples_num = len(img_names)
        print('total image num: ', self.samples_num)
        # 初始化self.imgs和self.labels，开辟对应大小空间和类型，与配置文件设置有关
        self.imgs = np.zeros((len(img_names), self.cfg['height'], self.cfg['width'], self.cfg['channel']), np.float32)
        self.labels = np.zeros((len(img_names), self.cfg['height'], self.cfg['width'], 1), np.float32)
        # cv2.imread读取后格式为unit8，所以遍历所有图片及标签进行读取图片并设置格式
        # 注：这里对标签进行了归一化处理，但并未对训练图片进行归一化处理，后续可考虑训练时对训练图片进行归一化看训练效果，同时要注意修改测试时同样处理方式
        for it in range(len(self.labels)):
            tp_img = cv2.imread(os.path.join(self.cfg['image_path'], img_names[it]))
            tp_label = cv2.imread(os.path.join(self.cfg['image_path'], label_names[it]), cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_GRAYSCALE加载一张灰度图
            self.imgs[it, :, :, :] = tp_img.astype(np.float32)
            self.labels[it, :, :, 0] = (tp_label/255).astype(np.float32)
        # 图像减去均值是为了让损失函数平滑收敛，但是这里的均值是直接读取配置文件
        # 若之后搭建训练集后可考虑计算下均值做相应改变
        self.imgs -= self.cfg['mean']
        print('images and labels reading done!')

    def batch_iterator(self, shuffle=False):
        '''
        根据batch进行迭代，利用shuffle进行打乱顺序，批次大小配置文件配置
        :param shuffle: 是否打乱
        :return: 单个训练图片和标签文件用于迭代
        '''
        batch_size = self.cfg['batch_size']
        num_examples = len(self.imgs)
        idx = list(range(num_examples))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, num_examples, batch_size):
            imgs = self.imgs[idx[i:min(i+batch_size, num_examples)], :, :, :]
            labels = self.labels[idx[i:min(i+batch_size, num_examples)], :, :, :]
            # print('batch_size: ', labels.shape[0])
            # yield是一个生成器generator，简单的说就是每次执行到yield就返回，下次又迭代进入时又从yield处继续，实现迭代
            yield imgs, labels


def arg_parser():
    '''
    GPU配置parser
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    args = parser.parse_args()
    return args


def sess_config(args=None):
    '''
    session的config配置，若没有GPU将直接使用CPU，不会报错
    :param args:
    :return: 配置config
    '''
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU 0
    config1 = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)
    return config1


if __name__ == "__main__":
    # 读取配置文件
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    args = arg_parser()     # 配置
    config = sess_config(args)   # session的config配置

    # 训练数据
    dataset = DataSet()
    # HED网络定义
    hed_class = HED(height=cfg['height'], width=cfg['width'], channel=cfg['channel'])
    sides = hed_class.vgg_hed()
    # 损失函数定义
    loss_class = HedLoss(sides)
    loss = loss_class.calc_loss()
    # 优化器，采用动态学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-5,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.1,
                                               staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    # summary记录，用于tensorboard可视化
    tf.summary.scalar(name='lr', tensor=learning_rate)
    hed_class.summary()
    loss_class.summary()
    merged_summary_op = tf.summary.merge_all()

    startTime = time()
    # 训练
    with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
        saver = tf.train.Saver()

        # session变量初始化
        sess.run(tf.global_variables_initializer())
        # 初始化HED网络权重（HED网络基于VGG16，直接使用VGG16权重初始化）
        hed_class.assign_init_weights(sess)

        # 断点续训
        ckpt_dir = cfg['model_weights_path']
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt != None:
            saver.restore(sess, ckpt)
            # sess.run(tf.assign(global_step, 0))
            print('saver restore finish')
        else:
            print("training from scratch")



        # 日志记录
        summary_writer = tf.summary.FileWriter(cfg['log_dir'], graph=sess.graph, flush_secs=15)

        step = 0    # 记录summary的step
        # 通过配置文件max_epochs最大迭代次数进行训练
        for epoch in range(1, cfg['max_epochs']+1):
            for imgs, labels in dataset.batch_iterator():   # 通过迭代获取训练图片和标签信息
                '''
                # 输出中间过程图片
                print("*************")
                print("*************")
                sides_show = sess.run([sides],feed_dict={hed_class.x: imgs})
                picture_batch1 = sides_show[0][0] #对应第一层输出的图片batch

                plt.subplot(331)
                outImage = sides_show[0][0]
                plt.imshow(outImage[0, :, :, 0])
                plt.subplot(332)
                outImage = sides_show[0][1]
                plt.imshow(outImage[0, :, :, 0])
                plt.subplot(333)
                outImage = sides_show[0][2]
                plt.imshow(outImage[0, :, :, 0])
                plt.subplot(334)
                outImage = sides_show[0][3]
                plt.imshow(outImage[0, :, :, 0])
                plt.subplot(335)
                outImage = sides_show[0][4]
                plt.imshow(outImage[0, :, :, 0])
                plt.subplot(336)
                outImage = sides_show[0][5]
                plt.imshow(outImage[0, :, :, 0])
                plt.show()

                print("*************")
                # print(sides_show)
                # print("*************")
                # print("*************")
                '''
                # 核心训练语句，利用训练图片和标签图片信息代入优化器进行训练，记录summary
                merged_summary, _ = sess.run([merged_summary_op, train_op],feed_dict={hed_class.x: imgs, loss_class.label: labels})
                if not (step % 1):
                    summary_writer.add_summary(merged_summary, global_step=step)
                    print('save a merged summary !')
                step += 1
                print('global_step:', sess.run(global_step), 'epoch: ', epoch)

            # 配置文件设置多少代输出一次模型
            if not epoch % cfg['snapshot_epochs']:
                saver.save(sess=sess, save_path=os.path.join(cfg['model_weights_path'], 'vgg16_hed'), global_step=epoch)
                print('save a snapshoot !')
        summary_writer.close()
        saver.save(sess=sess, save_path=os.path.join(cfg['model_weights_path'], 'vgg16_hed'), global_step=epoch)
        print('save final model')

    duration = time() - startTime
    print("train takes:", "{:.2f}".format(duration))


