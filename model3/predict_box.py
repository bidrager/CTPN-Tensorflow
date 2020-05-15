from model3.build_model import Model
import cv2
import model3.config  as cfg
import numpy as np
from model3.utils import *
import tensorflow as tf


def py_cpu_nms(dets,pro, thresh):
    """Pure Python NMS baseline."""
    scores = pro[:, -1]
    # inds = np.where(scores >= 0.9)[0]
    # scores=scores[inds]
    # dets=dets[inds]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# def connector(boxes):
#     bboxes=[]
#     while len(boxes)>0:
#         for
#
#
# with open(cfg.label_file,'r') as f:
#     all_data=json.load(f)

test_image_dir='../iamges/img_calligraphy_00269_bg.jpg'
sess=tf.Session()
model=Model(istrain=False,batch_size=cfg.batch_size)
model.model()
model.preict()
init=tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(cfg.saved_model)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print('SUCCESSED RELOAD')

image_datas=cv2.imread(test_image_dir)
height, width,c=image_datas.shape
image=[(image_datas-127.5)/127.5]
index,bais,probility=sess.run([model.target,model.anchor_transform,model.trust_pro],feed_dict={model.data:image})
index=np.reshape(index,[-1])
bais=np.reshape(bais,[-1,cfg.box_label_size])
probility=np.reshape(probility,[-1,2])
image_anchor = generate_single_image_anchors(height, width)
anchors=image_anchor[index]
bboxes=get_predict_box(anchors,bais)
bboxes=np.array(bboxes)
keep=py_cpu_nms(bboxes,probility,0.3)
bboxes=bboxes[keep]
inds_inside = np.where(
    (bboxes[:, 0] >= 0) &
    (bboxes[:, 1] >= 0) &
    (bboxes[:, 2] <= width) &
    (bboxes[:, 3] <= height))[0]
bboxes=bboxes[inds_inside]
for box in bboxes:
    cv2.rectangle(image_datas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
cv2.imwrite('../result/result.jpg',image_datas)
