from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(100)

data = np.rint(batch_xs[0]).astype(int)
label = np.rint(batch_ys[0]).astype(int)
pixels = data.reshape((28,28))

print(data)

print(str(pixels).replace(" ", ""))
print(label)

plt.imshow(pixels, cmap="gray")
plt.show()
