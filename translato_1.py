import tensorflow as tf
import numpy as np
import ReadingData as rd
from modules import multihead_attention

learning_rate=0.05
batch_size=10
max_samples=200
display_step=1
n_hidden=[80,80,80,80,80,80,80,80]
n_input=80
init_conc=np.zeros([batch_size,80]).astype('float32')
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.cast(tf.Variable(initial),tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.cast(tf.Variable(initial),tf.float32)

mel=rd.mel_dataset()
a,b=mel.get_next_batch()
a=np.array(a)
a_l=[a.shape[1],a.shape[2]]
b=np.array(b)

b_l=[b.shape[1],b.shape[2]]

n_steps=a.shape[1]
print('n_steps',n_steps)
x=tf.placeholder("float",[None,a.shape[1],n_input])
y=tf.placeholder("float",[None,b.shape[1],n_input])

batch_x,batch_y=mel.get_next_batch(batch_size)
w_fc=[]
b_fc=[]
w_fc0=weight_variable([a_l[0],b_l[0]])
w_fc.append(w_fc0)
b_fc0 = bias_variable([b_l[0]])
b_fc.append(b_fc0)
w_fc1=weight_variable([2*batch_size,batch_size])
w_fc.append(w_fc1)
b_fc1 = bias_variable([batch_size])
b_fc.append(b_fc1)

w_1_1=weight_variable([n_input,100])
b_1_1=bias_variable([100])
w_1_2=weight_variable([100,n_input])
b_1_2=bias_variable([n_input])

w_pre_1=weight_variable([n_input,32])
b_pre_1=bias_variable([32])
w_pre_2=weight_variable([32,n_input])
b_pre_2=bias_variable([n_input])

w_post_1=weight_variable([512*(a_l[0]-20),b_l[0]*n_input])
b_post_1=bias_variable([b_l[0]*n_input])
sess1=tf.Session()

def Stacked_BLSTM(x,w_fc,b_fc): #module 1
    x_s = tf.unstack(x, n_steps,1)

    lstm_fw_cell=[tf.contrib.rnn.BasicLSTMCell(size) for size in n_hidden]
    lstm_bw_cell=[tf.contrib.rnn.BasicLSTMCell(size) for size in n_hidden]
    stacked_lstm_fw=tf.contrib.rnn.MultiRNNCell(lstm_fw_cell)
    stacked_lstm_bw=tf.contrib.rnn.MultiRNNCell(lstm_bw_cell)
    outputs,_,_=tf.contrib.rnn.static_bidirectional_rnn(stacked_lstm_fw,stacked_lstm_bw,x_s,dtype=tf.float32)


    output_flat=tf.reshape(outputs,[-1,a_l[0]])

    output_fal = tf.nn.relu(tf.matmul(output_flat,w_fc[0]) + b_fc[0])
    output1=tf.reshape(output_fal,[-1,b_l[0],80])

    output_flat2=tf.reshape(output1,[-1,2*batch_size])
    output_fal2 = tf.nn.relu(tf.matmul(output_flat2,w_fc[1]) + b_fc[1])
    output_B=tf.reshape(output_fal2,[batch_size,b_l[0],80])
    return output_B

def Attention(output_B):
    output_A=multihead_attention(queries=output_B,keys=output_B,values=output_B,num_heads=4,dropout_rate=0.5,training=False,causality=False)
    return output_A
#The code needs to be modified

def Decoder(input_content,sess):
    #finish the lstm layer, the point is to change the dimensions
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=80, forget_bias=1.)
    input_content= tf.transpose(input_content, [1,0,2])
    X=tf.placeholder('float', [None,None,160])
    batch_size = tf.shape(input_content)[1]
    time_size=tf.shape(input_content)[0]
    initial_states = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    one_time=tf.concat([input_content[0,:,:], init_conc],1)
    one_time=tf.expand_dims(one_time,0)
    output_lstm,next_state = tf.nn.dynamic_rnn(inputs=X, cell=lstm_cell, dtype=tf.float32,initial_state=initial_states,time_major=True)
    output_lstm_flat=tf.reshape(output_lstm,[-1,n_input])
    output_lstm_fl1 = tf.matmul(output_lstm_flat,w_1_1) + b_1_1
    output_lstm_fl2 = tf.matmul(output_lstm_fl1,w_1_2) + b_1_2
    outputs=tf.reshape(output_lstm_fl2,[1,batch_size,80])
    #stop_token=tf.nn.sigmoid(output_linear_2_2)
    print("----------------------------finish the most pivotal work!------------------")
    for i in range(1,a_l[0]):
        next_input=tf.concat([input_content[i,:,:],outputs[-1,:,:]],1)
        next_input=tf.expand_dims(next_input,0)
        output_lstm,next_state = tf.nn.dynamic_rnn(inputs=next_input, cell=lstm_cell, dtype=tf.float32,initial_state=next_state,time_major=True)
        output_lstm_flat=tf.reshape(output_lstm,[-1,n_input])
        output_lstm_fl1 = tf.matmul(output_lstm_flat,w_1_1) + b_1_1
        output_lstm_fl2 = tf.matmul(output_lstm_fl1,w_1_2) + b_1_2
        outputs_part=tf.reshape(output_lstm_fl2,[-1,batch_size,80])
        outputs=tf.concat([outputs,outputs_part],0)
        pre_input_flat=tf.reshape(outputs_part,[-1,n_input])
        pre_input_fl1 = tf.matmul(pre_input_flat,w_pre_1) + b_pre_1
        pre_input_fl2 = tf.matmul(pre_input_fl1,w_pre_2) + b_pre_2
        pre_input=tf.reshape(pre_input_fl2,[-1,batch_size,80])
    #dimension needs to be changed
    output_sub= tf.transpose(outputs, [1,0,2])
    output_conv1 = tf.nn.tanh(tf.layers.conv1d(output_sub, filters=512, kernel_size=5))
    output_conv2 = tf.nn.tanh(tf.layers.conv1d(output_conv1, filters=512, kernel_size=5))
    output_conv3 = tf.nn.tanh(tf.layers.conv1d(output_conv2, filters=512, kernel_size=5))
    output_conv4 = tf.nn.tanh(tf.layers.conv1d(output_conv3, filters=512, kernel_size=5))
    output_conv5 = tf.layers.conv1d(output_conv4, filters=512, kernel_size=5)

    output_true_flat=tf.reshape(output_conv5,[-1,(a_l[0]-20)*512])
    output_true_fl1 = tf.matmul(output_true_flat,w_post_1) + b_post_1
    outputs_true=tf.reshape(output_true_fl1,[batch_size,b_l[0],80])
    return outputs_true
# the LSTMStateTuple is not able to be used as a placeholder

with tf.Session() as sess:
    pred1=Stacked_BLSTM(x,w_fc,b_fc)
    pred2=Attention(pred1)
    pred=Decoder(pred2,sess)
    pred_shape=tf.shape(pred)
    #cost=tf.reduce_mean(tf.losses.absolute_difference(predictions=pred,labels=y))
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #step=1
    sess.run(tf.global_variables_initializer())
    print(sess.run(pred_shape,feed_dict={x:batch_x,y:batch_y}))
    '''
    while step*batch_size<max_samples:
        batch_x,batch_y=mel.get_next_batch(batch_size)
        print(sess.run(pred_shape,feed_dict={x:batch_x,y:batch_y}))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step==0:
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("pred.shape:",sess.run(tf.shape(pred),feed_dict={x:batch_x,y:batch_y}))
            print("Iter" + str(step * batch_size) + ",Loss=",loss)
        step+=1
    '''
