# read MNIST data from http://yann.lecun.com/exdb/mnist/
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# data split in 3: 55k ex mnist.train, 10k mnist.train, 5k mnist.validation
# mnist.train.images = x (28 x 28 pixels => 784 array), minst.train.labels = y

# therefore, mnist.train.images = tensor (n-dim array: 55k x 784 array) | each pixel val [0,  1]
# mnist.train.labels (55k x 10) => use one-hot vectors


# train model using SOFTMAX regressions (y = softmax(Wx + b))
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784]) # create input placeholder
W = tf.Variable(tf.zeros([784, 10]))        # weights variable 
b = tf.Variable(tf.zeros([10]))             # bias variable
y = tf.matmul(x, W) + b                     # define y


y_ = tf.placeholder(tf.float32, [None, 10]) # define loss and optimizer

# The raw formulation of cross-entropy, can be numerically unstable
# So here we use tf.nn.softmax_cross_entropy_with_logits on the rawoutputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) # implement model, softmax and entropy

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #specify learning rate and entropy to optimize



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()     # initialize vars


# Training
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)                # select batch to run g.d.
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})     # run the defined train step

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # tf.argmax(y, 1) -> prediction; tf.argmax(y_, 1) -> actual; 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print("\nThe accuracy of the model on the test data is: " )
print(acc_val, "(", str(acc_val * 100) + "%", ")")