import tensorflow as tf

x = tf.Variable(0.)
with tf.GradientTape() as tape:
     y = 2 * x + 3

grad_of_y_wrt_x = tape.gradient(y, x)