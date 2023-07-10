# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import yaml
from hed_net import HED
import cv2
import argparse
import gc

'''
测试验证类：
设定合适的图像路径，恢复模型可生成结果图片及视频，对应路径为./data/s2.avi,./data/tb_black_img.png,./data/tb_gray_img.png
tensorboard 可视化：
tensorboard --logdir=D:\python\hed-tf-master\logs
'''


def arg_parser():
    '''
    定义gpu和图片路径
    :return: 配置args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    parser.add_argument('-img_path', type=str, required=False, default="")
    args = parser.parse_args()
    return args


def sess_config(args=None):
    '''
    定义session的config配置
    :param args: 配置信息，定义Gpu等
    :return: config配置
    '''
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU 0
    # config = tf.ConfigProto(log_device_placement=log_device_placement,
    #                         allow_soft_placement=allow_soft_placement,
    #                         gpu_options=gpu_options)
    config1 = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement)
    return config1


def img_pre_process(img, **kwargs):
    '''
    图片预处理
    :param img: 图片
    :param kwargs: 配置信息，如均值
    :return: 处理后的图片
    '''
    def stretch(bands, lower_percent=2, higher_percent=98, bits=8):
        if bits not in [8, 16]:
            print('error ! dest image must be 8bit or 16bits !')
            return
        # 创建一个0矩阵shape形同输入的bands
        out = np.zeros_like(bands, dtype=np.float32)
        n = bands.shape[2]
        for i in range(n):
            a = 0
            b = 1
            # numpy.percentile常用于处理离群数据点
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            if d-c == 0:
                out[:, :, i] = 0
                continue
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            # numpy.clip用于将数据元素限制在a,b之间，如t=[1,2,3,4,5],a=2,b=4,np.clip后为[2,2,3,4,4]
            out[:, :, i] = np.clip(t, a, b)
        if bits == 8:
            return out.astype(np.float32)*255
        else:
            return np.uint16(out.astype(np.float32)*65535)

    img = stretch(img)
    # 减去配置文件的均值
    img -= kwargs['mean']
    return img


def predict_big_map(img_path, out_shape=(448, 448), inner_shape=(224, 224), out_channel=1, pred_fun=None, **kwargs):
    """
    预测，生成对应的边缘检测图和视频，若图片不一致，会进行切割合成等操作输出固定大小的图片
    注：这样进行切割合成的图片对于进行全图边缘检测是可以的，但是对于特定需求边缘检测这样是不合适的。当然如果测试和训练图片大小一致，这个函数同样会输出等大合适的结果
    :param img_path: 图片路径
    :param out_shape: 输出图片大小
    :param inner_shape: 输入图片大小
    :param out_channel: 预测图片输出通道，通常为黑白图像，通道数1
    :param pred_fun: 前向计算模型，调用sess.run计算hed中间层，返回图片数据
    :return: 预测的图片结果
    """
    make_video = True   # 是否生成video文件

    image = cv2.imread(img_path, )  #读取图片
    # 如果图片只有二维，添加一维生成满足网络要求的占位符比如（?,224,224,1）
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        gc.collect()    # gc为垃圾回收

    # 以下大量代码为如果图片大小不满足网络占位符需求，则对图片进行拆分迭代分别计算每个子图片的
    pd_up_h, pd_lf_w = np.int64((np.array(out_shape)-np.array(inner_shape)) / 2)
    # print(image.shape)
    ori_shape = image.shape
    pd_bm_h = (out_shape[0]-pd_up_h) - (image.shape[0] % inner_shape[0])
    pd_rt_w = (out_shape[1]-pd_lf_w) - (image.shape[1] % inner_shape[1])
    it_h = np.int64(np.ceil(1.0*image.shape[0] / inner_shape[0]))
    it_w = np.int64(np.ceil(1.0*image.shape[1] / inner_shape[1]))
    image_pd = np.pad(image, ((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w), (0, 0)), mode='reflect').astype(np.float32)  # the image is default a color one
    # print(image_pd.shape)
    # print((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w))
    gc.collect()
    tp1 = np.array(inner_shape[0] - ori_shape[0] % inner_shape[0])
    tp2 = np.array(inner_shape[1] - ori_shape[1] % inner_shape[1])
    if ori_shape[0] % inner_shape[0] == 0:
        tp1 = 0
    if ori_shape[1] % inner_shape[0] == 0:
        tp2 = 0
    out_img = np.zeros((ori_shape[0]+tp1, ori_shape[1]+tp2, out_channel), np.float32)

    # video config #################################
    if make_video:
        fps = 24  # 视频帧率
        wd = 1360
        ht = int(1360*out_img.shape[0]/out_img.shape[1])
        # haha = np.zeros((ht, wd, 3), np.uint8)
        haha = cv2.resize(np.pad(image, ((0, tp1), (0, tp2), (0, 0)), mode='reflect'), (wd, ht), interpolation=cv2.INTER_LINEAR)
        video_writer = cv2.VideoWriter('./data/s2.avi',
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                       (wd, ht))   # isColor=False? (1360,480)为视频大小
    image = None  # release memory
    # main loop
    for ith in range(0, it_h):
        h_start = ith * inner_shape[0]
        count = 1
        for itw in range(0, it_w):
            w_start = itw*inner_shape[1]
            tp_img = image_pd[h_start:h_start+out_shape[0], w_start:w_start+out_shape[1], :]

            # image pre-process
            tp_img = img_pre_process(tp_img.copy(), **kwargs)
            # print('tp_img', tp_img.shape)

            tp_out = pred_fun(tp_img[np.newaxis, :])
            tp_out = np.squeeze(tp_out, axis=0)

            # image post-process
            # tp_out = post-process

            out_img[h_start:h_start+inner_shape[0], w_start:w_start+inner_shape[1], :] = tp_out[pd_up_h:pd_up_h+inner_shape[0], pd_lf_w:pd_lf_w+inner_shape[1], :]

            # write video ##########################
            if make_video:
                tp = cv2.resize(out_img[:, :, 0], (wd, ht), interpolation=cv2.INTER_LINEAR)
                # print(np.unique(tp))
                # xixi = np.uint8((tp > 0.5)*255)
                xixi = tp > 1e-5
                mimi = np.uint8(tp[xixi] * 255)
                haha[xixi, 0] = mimi
                haha[xixi, 1] = mimi
                haha[xixi, 2] = mimi
                video_writer.write(haha)

            print('haha!', h_start, w_start, count)
            count += 1
    if make_video:
        video_writer.release()
    return out_img[0:ori_shape[0], 0:ori_shape[1], :]


def predict_big_map_show(img_path, pred_fun=None, **kwargs):
    '''
    展示中间层图片
    注：由于没有类似predict_big_map切分图片的操作，这里对图片的大小严苛，需要形同训练图片大小才行，否则会报错，后续可增添相应的图片缩放等操作进行通用化处理。
    :param img_path: 图片路径
    :param pred_fun:前向计算模型，调用sess.run计算hed中间层，返回图片数据，可直接处理该数据进行可视化展示
    :param kwargs: 均值信息
    :return:无返回，调用该函数opencv弹窗显示中间图片
    '''
    image = cv2.imread(img_path, )
    # 如果图片只有二维，添加一维生成满足网络要求的占位符比如（?,224,224,1）
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        gc.collect()
    image = image.astype(np.float32)  # cv2.imread后类型为unit8，转类型处理
    # 图片前置处理
    tp_img = img_pre_process(image.copy(), **kwargs)
    # 图片预测，这里相当于pred_fun为一个函数传入，这里调用该函数，该函数已经使用session对传入的图片进行处理，输出结果直接为hed网络中6张图片的数据
    tp_out = pred_fun(tp_img[np.newaxis, :])
    print("***********")
    print("***********")
    # numpy.squeeze把维度为1的条目去除，
    img1 = np.squeeze((tp_out[0] * 255).astype(np.uint8))
    img2 = np.squeeze((tp_out[1] * 255).astype(np.uint8))
    img3 = np.squeeze((tp_out[2] * 255).astype(np.uint8))
    img4 = np.squeeze((tp_out[3] * 255).astype(np.uint8))
    img5 = np.squeeze((tp_out[4] * 255).astype(np.uint8))
    img6 = np.squeeze((tp_out[5] * 255).astype(np.uint8))
    # 图集
    output_img = np.hstack([img1, img2, img3, img4, img5, img6])
    # 展示多个
    cv2.namedWindow('ouput_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('ouput_image', output_img)
    cv2.waitKey(0)
    print("***********")
    print("***********")


if __name__ == '__main__':
    # 定义gpu和图片路径
    args = arg_parser()
    # 定义session的config配置
    config = sess_config(args)
    # 读取配置文件
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    # path = args.img_path
    # path = "data/dataset/train_data/IMG_1580.JPG"
    path = "data/dataset/IMG_2717_224.JPG"

    # 读取高、宽、通道，均值等
    height = cfg['height']
    width = cfg['width']
    channel = cfg['channel']
    mean = cfg['mean']

    # 定义session，HED网络
    sess = tf.Session(config=config)
    hed_class = HED(height=height, width=width, channel=channel)
    hed_class.vgg_hed()
    saver = tf.train.Saver()
    # 读取权重
    saver.restore(sess, 'data/weights/model_weights/vgg16_hed-120')

    '''
    # 如果这样调用中间图片可视化展示将导致图片非常尖锐不平滑
    sides = [hed_class.side1,hed_class.side2,hed_class.side3,hed_class.side4,hed_class.side5,hed_class.fused_side]
    predict_big_map_show(img_path=path, out_shape=(224, 224), inner_shape=(224, 224), out_channel=1,pred_fun=(lambda ipt: sess.run(sides, feed_dict={hed_class.x: ipt})), mean=cfg['mean'])
    '''
    # sigmoid处理后图片更加平滑
    sides = [tf.sigmoid(hed_class.side1),
             tf.sigmoid(hed_class.side2),
             tf.sigmoid(hed_class.side3),
             tf.sigmoid(hed_class.side4),
             tf.sigmoid(hed_class.side5),
             tf.sigmoid(hed_class.fused_side)]
    # 可视化展示
    predict_big_map_show(img_path=path,pred_fun=(lambda ipt: sess.run(sides, feed_dict={hed_class.x: ipt})), mean=cfg['mean'])

    # tf.add_n表示相加，这里的sides为相加后取均值实现图片融合
    sides = 1.0*tf.add_n(sides) / len(sides)
    # 图片预测
    output_img = predict_big_map(img_path=path, out_shape=(height, width), inner_shape=(height, width), out_channel=1,
                                 pred_fun=(lambda ipt: sess.run(sides, feed_dict={hed_class.x: ipt})), mean=cfg['mean'])

    # 去除维度为1的，实现图片转化
    output_img = np.squeeze((output_img*255).astype(np.uint8))
    cv2.imwrite('./data/tb_gray_img.png', output_img)
    cv2.imwrite('./data/tb_black_img.png', 255*(output_img > 127))
    sess.close()


