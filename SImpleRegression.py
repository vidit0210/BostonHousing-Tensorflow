import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X=(X-mean)/std
    return X

#Getting the Dataset
boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train,Y_train = boston.data[:,5],boston.target
X_train = normalize(X_train)
n_samples = len(X_train)

X=tf.placeholder(tf.float32,name="X")
Y=tf.placeholder(tf.float32,name="y")

b=tf.Variable(0.0)
w=tf.Variable(0.0)
Y_hat = X*w +b

loss = tf.square(Y-Y_hat,name="loss")
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(0.001,global_step=global_step,decay_steps=100000,decay_rate=0.95,staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()
total =[]

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('b1',sess.graph)
    #Train model for 100 epoch
    for i in range(100):
        total_loss=0
        for x,y in zip(X_train,Y_train):
            _,l = sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            total_loss+=l
            total.append(total_loss/n_samples)
        print('Epoch {0}:Loss {1}'.format(i,total_loss/n_samples))
    writer.close()
    b_value,w_value = sess.run([b,w])

Y_pred = X_train * w_value + b_value
print('Done')
plt.plot(X_train,Y_train,'bo',label='Real Data')
plt.plot(X_train,Y_pred,'r',label='Predicted Data')
plt.legend()
plt.show()
plt.plot(total)
plt.show()

