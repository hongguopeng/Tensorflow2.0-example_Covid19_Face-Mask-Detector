import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split


imagePaths = []
for files in os.listdir('./dataset'):
    for image in os.listdir('./dataset/{}'.format(files)):
        # os.path.splitext(image) => 獲取image的副檔名
        imagePaths.append('./dataset/{}/{}'.format(files , image))

# 獲取數據標簽
data , labels = [] , []
for imagePath in imagePaths:
    # 讀取image，並將image做resize
    image = load_img(imagePath, target_size = (224 , 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)

    if imagePath.split('/')[2] == 'without_mask':
        labels.append([0 , 1])
    if imagePath.split('/')[2] == 'with_mask':
        labels.append([1 , 0])

data = np.array(data , dtype = 'float32')
labels = np.array(labels).astype(np.float32)


# 訓練集與測試集切分
trainX , testX , trainY , testY =\
train_test_split(data , labels , test_size = 0.20 , random_state = 42)


# 對image做augmentation，防止overfitting
aug = ImageDataGenerator(rotation_range = 25,
                         width_shift_range = 0.15,
                         height_shift_range = 0.2 ,
                         shear_range = 0.2 ,
                         zoom_range = 0.15,
                         horizontal_flip = True ,
                         fill_mode = 'nearest')

# finetune with MobileNetV2
mobilenet = MobileNetV2(weights = 'imagenet' ,
                        include_top = False,
                        input_tensor = Input(shape = (224 , 224 , 3)))


class new_layer(tf.keras.Model):
    def __init__(self , basemodel):
        super().__init__()
        self.basemodel = basemodel
        self.maxpooling = tf.keras.layers.AveragePooling2D(pool_size = (7 , 7))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(units = 128 , activation = tf.nn.relu , name = 'layer_1')
        self.dense2 = tf.keras.layers.Dense(units = 2 , name = 'layer_2')
    
    def call(self , inputs , training = True):
        x = self.basemodel(inputs)            
        x = self.maxpooling(x)          
        x = self.flatten(x) 
        x = self.dense1(x)
        if training: x = self.dropout(x)
        x = self.dense2(x)    
        output = tf.nn.softmax(x , axis = 1)
        prediction = tf.math.log(tf.clip_by_value(output , 1e-8 , tf.reduce_max(output)))
        return prediction

mask_model = new_layer(mobilenet)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

@tf.function 
def train_step(x , y):
    with tf.GradientTape() as tape:
        prediction = mask_model(x , training = True)     
        cross_entropy_temp = -tf.reduce_sum(y * prediction , axis = 1)
        cross_entropy = tf.reduce_mean(cross_entropy_temp)
        correct = tf.equal(tf.math.argmax(prediction , 1) , tf.argmax(y , 1))
        correct = tf.cast(correct , tf.float32)
        accuracy = tf.reduce_mean(correct)
    var_list = [var for var in mask_model.trainable_variables if 'new_layer' in var.name]
    grads = tape.gradient(cross_entropy , var_list)   
    optimizer.apply_gradients(grads_and_vars = zip(grads , var_list))
    return cross_entropy , accuracy

@tf.function
def test_step(x , y):
    prediction = mask_model(x , training = False)     
    cross_entropy_temp = -tf.reduce_sum(y * prediction , axis = 1)
    cross_entropy = tf.reduce_mean(cross_entropy_temp)
    correct = tf.equal(tf.math.argmax(prediction , 1) , tf.argmax(y , 1))
    correct = tf.cast(correct , tf.float32)
    accuracy = tf.reduce_mean(correct)
    return cross_entropy , accuracy
    
for epoch_i in range(0 , 5):
    batches = 0
    for batch_i , (x_batch , y_batch) in enumerate(aug.flow(trainX , trainY , shuffle = True , batch_size = 32)):
        train_loss , train_acc = train_step(x_batch , y_batch)
             
        batches += 1
        if batches >= len(trainX) / 32: break
    
        if batch_i % 5 == 0:
            print('=' * 30)
            print('epoch_i : {}'.format(epoch_i))
            print('batch_i : {}'.format(batch_i))
            print('training_loss : {:.2f}'.format(train_loss.numpy()))
            print('training_accuracy : {:.2%}'.format(train_acc.numpy()))
       
    test_loss , test_acc = test_step(testX , testY)
    print('*' * 30)
    print('epoch_i : {}'.format(epoch_i))
    print('testing_accuracy : {:.2f}'.format(test_loss.numpy()))
    print('testing_accuracy : {:.2%}\n'.format(test_acc.numpy()))

# 模型存檔
tf.saved_model.save(mask_model , 'my_network')
sess = tf.compat.v1.Session()
tf.io.write_graph(sess.graph , './my_network' , 'mask_model.pbtxt')
restore_model = tf.saved_model.load('my_network')