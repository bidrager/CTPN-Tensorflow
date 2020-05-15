import numpy as np
import model3.config as cfg
import math


def get_iou(anchors,bbox):
    bbox=np.array(bbox)
    anchors_size=len(anchors)
    labels_size=len(bbox)
    iou_mat=np.zeros([anchors_size,labels_size])
    s_anchor=np.multiply((anchors[:,2]-anchors[:,0]),(anchors[:,3]-anchors[:,1]))
    s_bbox=np.multiply((bbox[:,2]-bbox[:,0]),(bbox[:,3]-bbox[:,1]))

    for i in range(labels_size):
        w=np.minimum(anchors[:,2],bbox[i,2])-np.maximum(anchors[:,0],bbox[i,0])
        h=np.minimum(anchors[:,3],bbox[i,3])-np.maximum(anchors[:,1],bbox[i,1])
        w[np.where(w<0)]=0
        h[np.where(h<0)]=0
        cross=np.multiply(w,h)
        s = np.add(s_anchor, s_bbox[i])
        iou_mat[:,i]= np.divide(cross, np.subtract(s, cross))
    return iou_mat

def generate_base_anchor():
    anchor_ws = cfg.anchor_size
    anchor_h=16
    anchors=[]
    y_ctr = (0 + anchor_h) * 0.5
    for w in anchor_ws:
        scaled_anchor=[]
        x_ctr = (0 + w) * 0.5
        scaled_anchor.append(x_ctr - w / 2)  # xmin
        scaled_anchor.append(y_ctr - anchor_h / 2)  # ymin
        scaled_anchor.append(x_ctr + w / 2)  # xmax
        scaled_anchor.append(y_ctr + anchor_h / 2)  # ymax
        anchors.append(scaled_anchor)
    return np.array(anchors)

def get_feature_shape(height,width):
    for i in range(4):
        height=math.ceil(height/2)
    for i in range(4):
        width=math.ceil(width/2)
    return int(height),int(width)

def generate_single_image_anchors(height,width):
    shift_x = np.arange(0, width,cfg.x_feat_stride)
    shift_y = np.arange(0, height,cfg.y_feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    anchors=generate_base_anchor()
    num_anchors=len(anchors)
    feature_h,feature_w=get_feature_shape(height,width)
    all_anchors = (np.reshape(anchors,(1, num_anchors, 4)) +
                   shifts.reshape((1, feature_h*feature_w, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((feature_h*feature_w*num_anchors, 4))
    return all_anchors

def get_predict_box(anchors,transforms):
    boxes=[]
    for i in range(len(anchors)):
        anchor_width = anchors[i][2] - anchors[i][0]
        anchor_height = anchors[i][3] - anchors[i][1]
        an_ctr_x = anchors[i][0] + 0.5 * anchor_width
        an_ctr_y = anchors[i][1] + 0.5 * anchor_height
        boxes_ctr_x = an_ctr_x - (transforms[i][0] * anchor_width)
        boxes_ctr_y = an_ctr_y - (transforms[i][1] * anchor_height)
        width=np.exp(transforms[i][2])*anchor_width
        boxes_x1 = int(boxes_ctr_x - width / 2)
        boxes_x2 = int(boxes_ctr_x + width / 2)
        boxes_y1 = int(boxes_ctr_y - anchor_height / 2)
        boxes_y2 = int(boxes_ctr_y + anchor_height / 2)
        boxes.append([boxes_x1,boxes_y1,boxes_x2,boxes_y2])
    return boxes

def get_rpn_label(anchor,bbox):
    anchor_width=anchor[2]-anchor[0]
    anchor_height=anchor[3]-anchor[1]
    an_ctr_x=anchor[0]+0.5*anchor_width
    an_ctr_y=anchor[1]+0.5*anchor_height

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    b_ctr_x = bbox[0] + 0.5 * bbox_width
    b_ctr_y = bbox[1] + 0.5 * bbox_height

    target_dx=(an_ctr_x-b_ctr_x)/anchor_width
    target_dy=(an_ctr_y-b_ctr_y)/anchor_height
    target_dh=np.log(bbox_width/anchor_width)
    return [target_dx,target_dy,target_dh]

def get_single_image_label(height,width,target):
    all_anchors=generate_single_image_anchors(height,width)

    num_labels=all_anchors.shape[0]
    cls_labels=np.ones([num_labels],dtype=np.uint8)*(-1)
    bbox_labels=np.zeros([num_labels,3])

    inds_inside = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] <= width) &  # width
        (all_anchors[:, 3] <= height)  # height
    )[0]
    anchors=all_anchors[inds_inside]
    # anchors_size=len(anchors)
    # labels_size=len(target)
    iou_mat=get_iou(anchors,target)
    # for i in range(anchors_size):
    #     for j in range(labels_size):
    #         iou=get_iou(anchors[i],target[j])
    #         iou_mat[i][j]=iou

    #为每一个bbox label 找到对应的iou最大的anchor label设为1 rpn_label设值
    argmax_overlap=iou_mat.argmax(axis=0) #每个bbox  iou最大的anchor  inds_inside id
    cls_labels[inds_inside[argmax_overlap]]=1
    for i in range(len(argmax_overlap)):
        bbox_labels[inds_inside[argmax_overlap[i]]]=get_rpn_label(anchors[argmax_overlap[i]],target[i])

    #为iou大于0.7的anchor设置正标签
    index=np.array([k for k in range(len(anchors)) if k not in argmax_overlap],dtype=np.int32)
    no_label_iou=iou_mat[index]
    anchor_max_overlap_index=no_label_iou.argmax(axis=1)
    anchor_max_overlap=no_label_iou[np.arange(len(no_label_iou)),anchor_max_overlap_index]
    posi=anchor_max_overlap>cfg.posi_bound
    nage=anchor_max_overlap<cfg.nage_bound
    posi_index=[i for i in range(len(posi)) if posi[i]]
    nage_index=[i for i in range(len(nage)) if nage[i]]
    cls_labels[inds_inside[index[posi_index]]]=1
    cls_labels[inds_inside[index[nage_index]]]=0
    for i in range(len(posi_index)):
        bbox_labels[inds_inside[index[i]]]=get_rpn_label(anchors[index[i]],target[anchor_max_overlap_index[i]])

    fg_inds=np.where(cls_labels==1)[0]
    if len(fg_inds)>cfg.num_fg:
        disable_inds=np.random.choice(fg_inds,size=len(fg_inds)-cfg.num_fg,replace=False)
        cls_labels[disable_inds]=-1

    bg_inds=np.where(cls_labels==0)[0]
    if len(bg_inds)>cfg.num_bg:
        disable_inds = np.random.choice(bg_inds, size=len(bg_inds) - cfg.num_bg, replace=False)
        cls_labels[disable_inds] = -1

    return cls_labels,bbox_labels



#测试anchor及label生产的是否准确

# label=[[0,0,16,16]]
# cls_labels,bbox_label=get_single_image_label(17,33,label)
#
# print(cls_labels)
# print('-----------------------')
# print(bbox_label[np.where(cls_labels==1)[0]])
# [ 1  1 -1 -1 -1 -1 -1  0  0  0  0 -1 -1 -1  0  0 -1 -1 -1 -1 -1  0  0 -1
#  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
# -----------------------
# [[ 0.          0.          0.        ]
#  [ 0.          0.125      -0.28768207]]
# import json
# import cv2
# all_image_dir='../data/big_word_image/img_calligraphy_00017_bg.jpg'
# with open('../process_data/anchor_box_label.json','r') as f:
#     all_data=json.load(f)
# label=all_data['img_calligraphy_00017_bg.jpg']
# image=cv2.imread(all_image_dir)
# height,width,c=image.shape
# for f in label:
#     cv2.rectangle(image, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), (0, 255, 0), 1)
#
# cv2.imwrite('a.jpg',image)
#
# cls_labels,bbox_label=get_single_image_label(height,width,label)
# index=np.where(cls_labels==1)[0]
# height_,width_=get_feature_shape(height,width)
# cls_labels=np.reshape(cls_labels,[height_,width_,6])
# heat_image=np.ones([int(height/16),int(width/16),3],dtype=np.uint8)*255
# for i in range(height_):
#     for k in range(width_):
#         for j in range(6):
#             if cls_labels[i,k,j]==1:
#                 heat_image[i,k,0:2]=0
#             #     break
#             #
#             elif cls_labels[i,k,j]==0:
#                 heat_image[i,k,0]=0
#
# cv2.imwrite('b.jpg',cv2.resize(heat_image,(width,height)))
# # #














