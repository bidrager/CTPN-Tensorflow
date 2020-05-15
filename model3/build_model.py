import tensorflow as tf
import model3.ops as ops
import json
import cv2
import os
import model3.config as cfg
from model3.utils import *


# 这里是预先处理好的训练数据（本程序未提供数据），运行predict_box.py这个地方报错就先注释了训练是需要打开的，下面的get_train_data注释同理见25行
# with open(cfg.label_file,'r') as f:
#     all_data=json.load(f)


all_image_dir=cfg.all_image_dir
image_dirs=os.listdir(all_image_dir)


def padding_image(image,bucket):
    height,width,channel=image.shape
    new_image=np.zeros(([bucket[1],bucket[0],3]))
    new_image[0:height,0:width,:]=image
    return new_image

# def get_train_data(batch_size):
#     image_dir=np.random.choice(image_dirs,p=cfg.P)
#     image_path=os.path.join(all_image_dir,image_dir)
#     images=os.listdir(image_path)
#     target_images=np.random.choice(images,batch_size,replace=False)
#     image_datas=[]
#     cls_labels=[]
#     bbox_labels=[]
#     heights=[]
#     widths=[]
#     for image in target_images:
#         image_name=os.path.join(image_path,image)
#         image_data=cv2.imread(image_name)
#         height,width,c=image_data.shape
#         heights.append(height)
#         widths.append(width)
#         image_data=(image_data-127.5)/127.5
#         image_datas.append(image_data)
#
#     datas = []
#     w=np.max(widths)
#     h=np.max(heights)
#     for name, image in zip(target_images,image_datas):
#         image_data = padding_image(image, [w, h])
#         label=all_data[name]
#         cls_label, bbox_label=get_single_image_label(h,w,label)
#         cls_labels.append(cls_label)
#         bbox_labels.append(bbox_label)
#         datas.append(image_data)
#     return datas,cls_labels,bbox_labels,image_dir

class Model():
    def __init__(self,istrain,batch_size):
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.label_class = tf.placeholder(tf.int32, shape=[None,None])
        self.label_bbox = tf.placeholder(tf.float32, shape=[None,None,cfg.box_label_size])
        self.istrain=istrain
        self.batch_size=batch_size

    def smooth_l1_dist(self, deltas, name='smooth_l1_dist'):
        with tf.name_scope(name=name):
            l2 = 0.5 * tf.square(deltas)
            l1 = tf.abs(deltas) - 0.5
            condition = tf.less(tf.abs(deltas), 1.0)
            re = tf.where(condition, l2, l1)
            return re

    def model(self):
        self.cnn_feature=ops.im2latex_cnn(self.data,self.istrain)
        self.bi_feature=ops.bi_LSTM(self.cnn_feature,name='bi_LSTM')

        self.class_feature=ops.liner(self.bi_feature,name='class_liner',
                                     output_size=cfg.anchor_num*cfg.class_num)
        self.regresion_feature = ops.liner(self.bi_feature, name='regresion_liner',
                                           output_size=cfg.anchor_num * cfg.box_label_size)
        # liner output shape N,H,W,(anchor-num*class_num or anchor-num*box_label_size)

    def build_loss(self):

        self.rpn_cls_score=tf.reshape(self.class_feature,[-1,2])
        self.rpn_label=tf.reshape(self.label_class,[-1])

        self.rpn_keep=tf.where(tf.not_equal(self.rpn_label,-1))
        self.fg_keep=tf.where(tf.equal(self.rpn_label,1))

        self.rpn_cls_score=tf.reshape(tf.gather(self.rpn_cls_score,self.rpn_keep),[-1,2])
        self.rpn_label=tf.reshape(tf.gather(self.rpn_label,self.rpn_keep),[-1])

        self.rpn_cross_entropy_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.rpn_label,logits=self.rpn_cls_score)
        self.rpn_cross_entropy_loss=tf.reduce_mean(self.rpn_cross_entropy_loss)

        self.rpn_bbox_pre=tf.reshape(self.regresion_feature,[-1,cfg.box_label_size])
        self.rpn_bbox_label=tf.reshape(self.label_bbox,[-1,cfg.box_label_size])

        self.rpn_bbox_pre=tf.reshape(tf.gather(self.rpn_bbox_pre,self.fg_keep),[-1,cfg.box_label_size])
        self.rpn_bbox_label=tf.reshape(tf.gather(self.rpn_bbox_label,self.fg_keep),[-1,cfg.box_label_size])

        self.rpn_bbox_loss=tf.reduce_sum(self.smooth_l1_dist((self.rpn_bbox_pre-self.rpn_bbox_label)),reduction_indices=1)
        self.rpn_bbox_loss=tf.reduce_mean(self.rpn_bbox_loss)

        self.loss=self.rpn_cross_entropy_loss+self.rpn_bbox_loss


    def train(self):
        self.optimizer=tf.train.AdamOptimizer(learning_rate=cfg.learning_rate)
        self.gvs=self.optimizer.compute_gradients(self.loss)
        self.clip_gvs=[(tf.clip_by_value(grad,-5.,5.), var) for grad, var in self.gvs]
        self.train_step=self.optimizer.apply_gradients(self.clip_gvs)


    def preict(self):
        self.probility = tf.nn.softmax(tf.reshape(self.class_feature,[-1,2]))
        self.class_pre=tf.argmax(self.probility,axis=1)
        # self.class_pre = tf.reshape(self.class_pre,[-1])
        self.target=tf.where(tf.equal(self.class_pre,1)) #预测为1的所有index
        self.anchor_transform = tf.reshape(self.regresion_feature, [-1, cfg.box_label_size])
        self.trust_pro=tf.gather(tf.reshape(self.probility,[-1,2]),self.target)#预测为1的所有anchor label
        self.anchor_transform=tf.gather(self.anchor_transform,self.target)#预测为1的所有 anchor bbox偏移量


