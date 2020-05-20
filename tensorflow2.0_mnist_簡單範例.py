import tensorflow as tf
# https://github.com/tensorflow/tensorflow
# 下載上述github網址中的內容，接著找到tensorflow/examples中的tutorials
# 再把整個資料夾複製到tensorflow_core\examples中
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data' , one_hot = True)
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = 100 , activation = tf.nn.relu , name = 'layer_1')
        self.dense2 = tf.keras.layers.Dense(units = 10 , name = 'layer_2')

    def call(self , inputs):             # [batch_size , 784]
        x = self.dense1(inputs)          # [batch_size , 100]
        x = self.dense2(x)               # [batch_size , 10]
        output = tf.nn.softmax(x , axis = 1)
        prediction = tf.math.log(tf.clip_by_value(output , 1e-8 , tf.reduce_max(output)))
        return prediction

classfier = MLP()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

loss_his = []
for batch_i in range(0 , 2000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    
    with tf.GradientTape() as tape:
        prediction = classfier(batch_xs)     
        cross_entropy_temp = -tf.reduce_sum(batch_ys * prediction , axis = 1)
        cross_entropy = tf.reduce_mean(cross_entropy_temp)
        
        correct = tf.equal(tf.math.argmax(prediction , 1) , tf.argmax(batch_ys , 1))
        correct = tf.cast(correct , tf.float32)
        accuracy = tf.reduce_mean(correct)
    
    grads = tape.gradient(cross_entropy , classfier.trainable_variables)   
    optimizer.apply_gradients(grads_and_vars = zip(grads , classfier.trainable_variables))
    print('=' * 30)
    print('batch_i : {}'.format(batch_i))
    print('training_loss : {:.2f}'.format(cross_entropy))
    print('training_accuracy : {:.2%}'.format(accuracy))
    loss_his.append(cross_entropy)
    
plt.plot(loss_his)   