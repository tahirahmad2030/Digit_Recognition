#This work creates CNN and learns to recognise digits from images. The dataset used is MNIST  
#Data is to be downloaded and extracted manually from http://yann.lecun.com/exdb/mnist/

import os
import struct
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

path = "." 
fname_img = os.path.join(path,'mnist_data/train-images.idx3-ubyte')# /home/dar/Project_SVHN/capstone/mnist_data
fname_lbl = os.path.join(path, 'mnist_data/train-labels.idx1-ubyte')

tname_img = os.path.join(path,'mnist_data/t10k-images.idx3-ubyte')# /home/dar/Project_SVHN/capstone/mnist_data
tname_lbl = os.path.join(path, 'mnist_data/t10k-labels.idx1-ubyte')


# Training data loading
with open(fname_lbl, 'rb') as flbl:
     magic, num = struct.unpack(">II", flbl.read(8))
     lbl = np.fromfile(flbl, dtype=np.int8)

with open(fname_img, 'rb') as fimg:
     magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
     img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

#Testing Data loading
with open(tname_lbl, 'rb') as flbl:
     magic, num = struct.unpack(">II", flbl.read(8))
     t_lbl = np.fromfile(flbl, dtype=np.int8)

with open(tname_img, 'rb') as fimg:
     magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
     t_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(t_lbl), rows, cols)

print (img.shape)
print (t_img.shape)
#For one hot encoding
def one_hot_encoding(labels):
    map = [0,1,2,3,4,5,6,7,8,9]
    lb = preprocessing.LabelBinarizer()
    lb.fit(map)
    lb.classes_
    return lb.transform(labels)

#Change shape of images
def reformat_images(images):
    return np.reshape(images, [-1,28*28])

test_labels = one_hot_encoding(t_lbl)
test_images = reformat_images(t_img)
print test_labels.shape
print test_images.shape


train_labels = one_hot_encoding(lbl)
train_data = reformat_images(img)

def get_next(step, batch):
    offset = (step * batch) % (train_labels.shape[0] - batch)
    batch_data = train_data[offset:(offset + batch), :]
    batch_labels = train_labels[offset:(offset + batch)]
    return batch_data, batch_labels


#Model Creation
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) 
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Hidden
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels= y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


for i in range(100):
  batch = get_next(i,50)
  if i%1 == 0:
    train_accuracy = accuracy.eval(feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

# Accuracy Testing
print("test accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_:test_labels , keep_prob: 1.0}))


