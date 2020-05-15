
posi_bound=0.7
nage_bound=0.3
x_feat_stride=16
y_feat_stride=16
all_image_dir='../data/'
label_file='../process_data/anchor_box_label.json'
conv_base_filter=32
rnn_size=128
bi_LSTM_out=256
class_num=2
anchor_height=16
batch_size=8
learning_rate=0.00004
rpn_conv_out_size=conv_base_filter*8
num_fg=128
num_bg=128
box_label_size=3
anchor_size=[16,24,38,54,80,140]
anchor_num=len(anchor_size)
epoch=24
start_step=0
end_step=int(50000*epoch/batch_size)+1
save_step=int(50000/batch_size)+1

saved_model='../save_model3/'
P=[0.13,0.35,0.34,0.18]