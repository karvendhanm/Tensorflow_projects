import tensorflow as tf

tensor_zeros = tf.zeros(shape= (128, 128))
print(tensor_zeros)

tensor_ones = tf.ones(shape= (128, 128))

x = tf.random.normal(shape= (3, 1), mean= 0., stddev=1.)
print(x)

x = tf.random.uniform(shape= (3, 1), minval= 0., maxval=1.)
print(x)

x = tf.ones(shape= (2, 2))
x[0, 0] = 0.

# mutable state tensor.
initial_value = tf.random.uniform((3, 1))
v = tf.Variable(initial_value=initial_value)
print(v)

v.assign(tf.ones((3, 1)))

# assigning value to the subset of the tensor
v[0, 0].assign(3.)

v.assign_add(tf.ones((3, 1)))
v.assign_sub(tf.ones((3, 1)))

print(v)

# doing math in tensorflow:
a = tf.ones((2, 2))
print(a)

b = tf.square(a)
print(b)

c = tf.sqrt(a)
print(c)

d = b + c
print(d)

e = tf.matmul(a, d) # matrix multiplication(matmul) is a dot product
print(e)

e *= d
print(e)  # this is element-wise multiplication


# gradient tape
input_var = tf.Variable(initial_value=3.)
input_var.shape

with tf.GradientTape() as tape:
    result = tf.sqrt(input_var)
gradient = tape.gradient(result, input_var)

# gradient tape watches only trainable variables by default. tf.Variable(initial_value)
# if we need the gradient tape to watch any other tensor, we need to invoke watch
input_var = tf.constant(value=3.)
with tf.GradientTape() as tape:
    tape.watch(input_var)
    result = tf.sqrt(input_var)
gradient = tape.gradient(result, input_var)

# second-order gradient
time = tf.Variable(initial_value = 0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)


