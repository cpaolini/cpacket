# python3.9 ResNet50.py
#
#https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
#https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl
#
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
os.environ['TF_CPP_VMODULE'] = "collective_util=1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import logging
import tensorflow as tf
import tensorflow.keras as K
import json
import tf_keras as keras

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
#from keras_applications.resnet import ResNet50

#from keras.applications.resnet50 import ResNet50
from keras_applications.resnet import preprocess_input, decode_predictions
#from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.platform import build_info as tf_build_info
import numpy as np

class CustomModel(keras.Sequential):
  def train_step(self, data):
    images, labels = data
    print("in train_step")
    if isinstance(images, tf.distribute.DistributedValues):
      # Execute the training step on each replica
      loss = self.distributed_train_step(images, labels)
      # Gather the results from all replicas
      return {'loss': tf.distribute.strategy.gather(loss, axis=0)}
    else:
      with tf.GradientTape() as tape:
        predictions = self(images, training=True)
        loss = self.compute_loss(labels, predictions)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss': loss}

      @tf.function
      def distributed_train_step(self, images, labels):
        print("in distributed_train_step")
        per_replica_losses = tf.distribute.get_strategy().run(self.train_step_for_single_replica, args=(images, labels))
        return per_replica_losses

      def train_step_for_single_replica(self, images, labels):
        print("in train_step_for_single_replica")
        with tf.GradientTape() as tape:
          predictions = self(images, training=True)
          loss = self.compute_loss(labels, predictions)

          trainable_vars = self.trainable_variables
          gradients = tape.gradient(loss, trainable_vars)
          self.optimizer.apply_gradients(zip(gradients, trainable_vars))
          return loss

def CIFAR_dataset(batch_size):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  print((x_train.shape, y_train.shape))
  x_train, y_train = preprocess_data(x_train, y_train)
  x_test, y_test = preprocess_data(x_test, y_test)
    
#    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  #x_train = x_train / np.float32(255)
  #y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000)
  return train_dataset

def dataset_fn(global_batch_size, input_context):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = CIFAR_dataset(batch_size)
  dataset = dataset.shard(input_context.num_input_pipelines,
                          input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  return dataset

def preprocess_data(X, Y):
    """
    Pre-processes the data for your model.

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
      where m is the number of data points.
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X.

    Returns: X_p, Y_p.
    X_p is a numpy.ndarray containing the preprocessed X.
    Y_p is a numpy.ndarray containing the preprocessed Y.
    """
    X_p = tf.keras.applications.resnet50.preprocess_input(X)
    Y_p = tf.keras.utils.to_categorical(Y, 10)
    return X_p, Y_p

#"worker": ["130.191.49.239:50000", "130.191.49.91:50000"]
#"worker": ["130.191.49.239:50000", "130.191.49.65:50000"]
def train_task(index, batchSize):

    print(f'train_task: index={index}, batchSize={batchSize}')

    tf_config = {
        'cluster': {
            "worker": ["130.191.49.239:50000", "130.191.49.91:50000", "130.191.49.65:50000"]
        },
        'task': {'type': 'worker', 'index': index}
    }
    
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    per_worker_batch_size = batchSize
    num_workers = len(tf_config['cluster']['worker'])
    global_batch_size = per_worker_batch_size * num_workers
    print(f'global_batch_size: {global_batch_size}')
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.RING)
    #communication = tf.distribute.experimental.CollectiveCommunication.RING
    #options = tf.distribute.experimental.CommunicationOptions(implementation=communication)
#    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=options)

    #print(f' * 2 * ')
    
    with strategy.scope():
      #multi_worker_dataset = CIFAR_dataset(batch_size=per_worker_batch_size)
      
      #multi_worker_dataset = strategy.distribute_datasets_from_function(
      #  lambda input_context: dataset_fn(global_batch_size, input_context))
      #        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
      #        print((x_train.shape, y_train.shape))
      #        x_train, y_train = preprocess_data(x_train, y_train)
      #        x_test, y_test = preprocess_data(x_test, y_test)
      #print((x_train.shape, y_train.shape))
      print("tf.keras.datasets.cifar10.load_data")
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
      #        print((x_train.shape, y_train.shape))
      x_train, y_train = preprocess_data(x_train, y_train)
      #        x_test, y_test = preprocess_data(x_test, y_test)
      #print((x_train.shape, y_train.shape))
      train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

      print("K.Input")
      input_t = K.Input(shape=(32,32,3))

      print("K.Input 2")
      res_model = ResNet50(include_top=False, weights="imagenet", input_tensor=input_t)

      print("K.Input 3")
      for layer in res_model.layers[:143]:
        layer.trainable = False
      print("K.Input 4")
      for i, layer in enumerate(res_model.layers):
        print(i, layer.name, "=", layer.trainable)
      print("K.Input 5")
      to_res = (32, 32)
      print("K.Input 6")        
      model = K.models.Sequential()
      #model = CustomModel()
      model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
      model.add(res_model)
      model.add(K.layers.Flatten())
      model.add(K.layers.BatchNormalization())
      model.add(K.layers.Dense(256, activation='relu'))
      model.add(K.layers.Dropout(0.5))
      model.add(K.layers.BatchNormalization())
      model.add(K.layers.Dense(128, activation='relu'))
      model.add(K.layers.Dropout(0.5))
      model.add(K.layers.BatchNormalization())
      model.add(K.layers.Dense(64, activation='relu'))
      model.add(K.layers.Dropout(0.5))
      model.add(K.layers.BatchNormalization())
      model.add(K.layers.Dense(10, activation='softmax'))
      print("K.callbacks.ModelCheckpoint")
      check_point = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                                monitor="val_acc",
                                                mode="max",
                                                save_best_only=True,
      )
      print("model.compile")
      model.compile(loss='categorical_crossentropy',
                      optimizer=K.optimizers.RMSprop(learning_rate=2e-5),
                      metrics=['accuracy'])

      #        multi_worker_model.fit(multi_worker_dataset, epochs=2, steps_per_epoch=20)

      #        history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1,
      #                            validation_data=(x_test, y_test),
      #                            callbacks=[check_point])

      #history = model.fit(multi_worker_dataset, epochs=2, verbose=1, callbacks=[check_point], steps_per_epoch=50)

      #                          validation_data=(x_test, y_test),
      print("model.fit")
      history = model.fit(train_data.repeat(), batch_size=batchSize, epochs=1, verbose=2, steps_per_epoch=1)
      #history = model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
      print("model.fit done")
      #model.summary()
      #model.save("cifar10.h5")


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass      

print(tf.__version__)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#print("CUDA Version:", tf_build_info.cuda_version_number)
#print("cuDNN Version:", tf_build_info.cudnn_version_number)

trainTaskIndex = sys.argv[1]
batchSize = 256 # int(sys.argv[2])

train_task(trainTaskIndex, batchSize)
