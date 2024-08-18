import tensorflow as tf
import tensorflow_datasets as tfds
import utils



class Block(tf.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        
        
        for i in range(repetitions):
             
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(self.filters, self.kernel_size, activation='relu', padding = 'same')
        
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size= pool_size, strides=strides, padding = 'same')
  
    def call(self, inputs):
        conv2D_0 = vars(self)['conv2D_0']
        x = conv2D_0(inputs)

        
        for i in range(1,self.repetitions):
            conv2D_i = vars(self)[f'conv2D_{i}']
            x = conv2D_i(x)
        max_pool = self.max_pool
        
        return max_pool






class MyVGG(tf.keras.Model):

    def __init__(self, num_classes):
        super(MyVGG, self).__init__()

       
        self.block_a = Block(64,3,2)
        self.block_b = Block(128,3,2)
        self.block_c = Block(256,3,3)
        self.block_d = Block(512,3,3)
        self.block_e = Block(512,3,3)
        self.flatten = tf.keras.layers.Flatten()
 
        self.fc = tf.keras.layers.Dense(256, activation = 'relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
       
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten
        x = self.fc
        x = self.classifier
        return x


dataset = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, data_dir='data/')


vgg = MyVGG(num_classes=2)


vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def preprocess(features):
     image = tf.image.resize(features['image'], (224, 224))
     return tf.cast(image, tf.float32) / 255., features['label']


dataset = dataset.map(preprocess).batch(32)


vgg.fit(dataset, epochs=10)

