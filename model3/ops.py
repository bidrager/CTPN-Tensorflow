import tensorflow as tf
import  model3.config as cfg


def norm_layer(name,x,train,eps=1e-5,decay=0.9):
    with tf.name_scope(name):
        param_shape=x.get_shape().as_list()[-1:]
        scale = tf.get_variable(name+'_scale',param_shape, initializer=tf.constant_initializer(1.))
        offset = tf.get_variable(name+'_offset', param_shape,initializer=tf.constant_initializer(0.0))
        moving_mean=tf.get_variable(name+'_mean',param_shape,initializer=tf.zeros_initializer,trainable=False)
        moving_variance=tf.get_variable(name+'_variance',param_shape,initializer=tf.zeros_initializer,trainable=False)
        mean, var = tf.nn.moments(x, axes=[0,1,2], name='moments')
        train_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        train_var_op = tf.assign(moving_variance, moving_variance * decay + var * (1 - decay))
        if train:
            with tf.control_dependencies([train_mean_op,train_var_op]):
                return tf.nn.batch_normalization(x,mean,var,offset,scale,eps)
        else:
            return tf.nn.batch_normalization(x,moving_mean,moving_variance,offset,scale,eps)

def create_weight(shape,name):
    w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32),
                           name=name,shape=shape)
    return w
def create_bias(shape,name):
    b =tf.get_variable(initializer=tf.constant_initializer(value=0,dtype=tf.float32),
                           name='bias_'+name,shape=shape)
    return b

def conv2d(input,name,filter_deep,filer_num,train):
    with tf.name_scope(name):
        conv1_w=create_weight(shape=[3,3,filter_deep,filer_num],name=name+'_w')
        conv1_b=create_bias([filer_num],name=name+'_b')
        conv1=tf.nn.conv2d(input,conv1_w,strides=[1,1,1,1],padding='SAME')
        conv=tf.nn.bias_add(conv1,conv1_b)
        conv=norm_layer(name+'_batch_norm',conv,train)
        return conv

def im2latex_cnn(X,train):
    with tf.name_scope('im2latex_cnn'):
        X = tf.nn.relu(conv2d( X, 'conv1',3, cfg.conv_base_filter,train))
        X = tf.nn.max_pool(name='pool1', value=X, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        X = tf.nn.relu(conv2d(X, 'conv2', cfg.conv_base_filter,cfg.conv_base_filter*2,train))
        X = tf.nn.max_pool(name='pool2',value= X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        X = tf.nn.relu(conv2d(X,'conv3', cfg.conv_base_filter*2,cfg.conv_base_filter*2,train))
        X = tf.nn.relu(conv2d(X, 'conv4', cfg.conv_base_filter*2,cfg.conv_base_filter*4,train))
        X = tf.nn.max_pool(name='pool4',value= X, ksize=[1,2,2,1], strides=[1, 2, 2, 1],padding='SAME')

        X = tf.nn.relu(conv2d(X, 'conv5', cfg.conv_base_filter * 4, cfg.conv_base_filter * 4,train))
        X = tf.nn.relu(conv2d(X, 'conv6', cfg.conv_base_filter * 4, cfg.conv_base_filter * 8,train))
        X = tf.nn.max_pool(name='pool6', value=X, ksize=[1,2,2,1], strides=[1, 2, 2, 1],padding='SAME')

        X = tf.nn.relu(conv2d(X, 'conv7', cfg.conv_base_filter * 8, cfg.conv_base_filter * 8,train))
        return X


def bi_LSTM(img,name):
    with tf.name_scope(name):
        img=tf.transpose(img,[0,2,1,3])
        shape = tf.shape(img)
        N, W, H, C = shape[0], shape[1], shape[2], shape[3]
        img = tf.reshape(img, [N * W, H, C])
        img.set_shape([None, None, cfg.rpn_conv_out_size])

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(cfg.rnn_size)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(cfg.rnn_size)
        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img,
                                                               dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)
        lstm_out = tf.reshape(lstm_out, [N * W * H, 2 * cfg.rnn_size])

        weights = create_weight(name='Bi_LSTM_out_w', shape=[2 * cfg.rnn_size, cfg.bi_LSTM_out])
        biases =create_bias(name='Bi_LSTM_out_b', shape=[cfg.bi_LSTM_out])
        outputs = tf.nn.bias_add(tf.matmul(lstm_out, weights) , biases)
        outputs = tf.reshape(outputs, [N, W, H, cfg.bi_LSTM_out])
        outputs = tf.transpose(outputs,[0,2,1,3])
        return outputs

def liner(input,name,output_size):
    with tf.variable_scope(name):
        shape = tf.shape(input)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        input = tf.reshape(input, [N * H * W, C])

        weights = create_weight(name='out_w', shape=[cfg.bi_LSTM_out,output_size ])
        biases = create_bias(name='out_b', shape=[output_size])
        outputs = tf.nn.bias_add(tf.matmul(input, weights), biases)
        _O = tf.reshape(outputs, [N, H, W, output_size])
        return _O




def regularization(vars,weight_decay=1e-8):
    for var in vars:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(weight_decay)(var))

    regular_loss = tf.add_n(tf.get_collection("losses"))
    return regular_loss

