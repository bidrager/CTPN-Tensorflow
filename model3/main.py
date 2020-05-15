from model3.build_model import *



sess=tf.Session()
model=Model(istrain=True,batch_size=cfg.batch_size)
model.model()
model.build_loss()
model.train()
init=tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(cfg.saved_model)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
for step in range(cfg.start_step,cfg.end_step):
    image_datas, cls_labels, bbox_labels, image_dir_ = get_train_data(model.batch_size)
    sess.run(model.train_step, feed_dict={model.data: image_datas,
                                            model.label_class: cls_labels, model.label_bbox: bbox_labels})
    if step % 100 == 0:
        los = sess.run(model.loss, feed_dict={model.data: image_datas,
                                            model.label_class: cls_labels, model.label_bbox: bbox_labels})
        print(image_dir_,los)
    if step % cfg.save_step == 0 and step > 1:
        cfg.learning_rate = cfg.learning_rate * 0.8
        saver.save(sess, cfg.saved_model + 'location_ctpn', global_step=step)
