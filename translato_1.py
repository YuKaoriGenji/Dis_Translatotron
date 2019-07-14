import t
ensorflow as tf
import numpy as np
import ReadingData as rd
from modules import multihead_attention

learning_rate=0.05
batch_size=10
max_samples=200
display_step=1
n_hidden=80
n_input=80
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
x=tf.placeholder("float",[None,a.shape[1],n_hidden])
y=tf.placeholder("float",[None,b.shape[1],n_hidden])

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

sess1=tf.Session()

def Stacked_BLSTM(x,w_fc,b_fc): #module 1
    x_s = tf.unstack(x, n_steps,1)

    lstm_fw_cell=[tf.contrib.rnn.BasicLSTMCell(size,forget_bias=1.0) for size in n_hidden]
    lstm_bw_cell=[tf.contrib.rnn.BasicLSTMCell(size,forget_bias=1.0) for size in n_hidden]
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
    output_A=multihead_attenntion(queries=output_B,key=output_B,values=output_B,num_heads=4,dropout_rate=0.5,training=False,causality=False)
    return output_A
#The code needs to be modified

def Decoder(input_content):
    #dinish the lstm layer, the point is to change
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=80, forget_bias=1.)
    input_content= tf.transpose(input_content, [1,0,2])
    X=tf.placeholder('float', [None,None,160])
    batch_size = tf.shape(input_content)[1]
    time_size=tf.shape(input_content)[0]
    init_conc=np.zeros(batch_size,80)
    init_conc=tf.convert_to_tensor(init_conc)
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    output_lstm, last_states = tf.nn.dynamic_rnn(inputs=X, cell=lstm_cell, dtype=tf.float32,initial_state=initial_states,time_major=True)

    output_linear_1_1 = tf.contrib.layers.fully_connected(output_lstm,100, activation_fn=tf.tanh)
    output_linear_1_2 = tf.contrib.layers.fully_connected(output_1_1,80, activation_fn=tf.tanh)


    # Whether it is needed or not I'm not sure
    output_linear_2_1 = tf.contrib.layers.fully_connected(output_lstm,10, activation_fn=tf.tanh)
    output_linear_2_2 = tf.contrib.layers.fully_connected(output_2_1,1, activation_fn=tf.tanh)
    stop_token=tf.nn.sigmoid(output_linear_2_2)

    fetches = {'final_state': last_states,
           'output': output_output_linear_1_2}
    one_time=tf.concat([input_content[0], init_conc], 2)
    feed_dict = {X:one_time}
    eval_out=fetches.eval(feed_dict)
    output_sub = [eval_out['prediction']]
    for i in range(1, seq_length):
        next_input=tf.concat([input_content[i],outputs[-1]], 2)
        feed_dict = {X: next_input,initial_state: next_state}
        eval_out = sess.run(fetches, feed_dict)
        output_sub.append(eval_out['output'])
        next_state = eval_out['final_state']
    #dimension needs to be changed
    output_sub= tf.transpose(outputs, [1,0,2])
    output_conv1 = tf.nn.tanh(tf.layers.conv1d(output_sub, filters=512, kernel_size=5))
    output_conv2 = tf.nn.tanh(tf.layers.conv1d(output_conv1, filters=512, kernel_size=5))
    output_conv3 = tf.nn.tanh(tf.layers.conv1d(output_conv2, filters=512, kernel_size=5))
    output_conv4 = tf.nn.tanh(tf.layers.conv1d(output_conv3, filters=512, kernel_size=5))
    output_conv5 = tf.layers.conv1d(output_conv4, filters=512, kernel_size=5)
    return output_conv5





pred=Stacked_BLSTM(x,w_fc,b_fc)
pred=Attention(pred)
pred=Decoder(pred,sess)
pred_shape=tf.shape(pred)
cost=tf.reduce_mean(tf.losses.absolute_difference(predictions=pred,labels=y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=1
    while step*batch_size<max_samples:
        batch_x,batch_y=mel.get_next_batch(batch_size)
        print(sess.run(pred_shape,feed_dict={x:batch_x,y:batch_y}))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step==0:
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("pred.shape:",sess.run(tf.shape(pred),feed_dict={x:batch_x,y:batch_y}))
            print("Iter" + str(step * batch_size) + ",Loss=",loss)
        step+=1
