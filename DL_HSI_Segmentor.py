# ---------------- DL-based segmentation (Tensorflow) -----------------------------
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


class CAE3D(object ):
     def __init__(self,
                  nbInputPatterns,
                  blk_size,
                  drop_rate,
                  drop_seed,
                  encoded_dim,
                  nFreqBands,
                  data_format,
                  active_function,
                  batch_sz = 20,
                  modelFileName = None):

          gpus = tf.config.experimental.list_physical_devices('GPU')
          tf.config.experimental.set_memory_growth(gpus[0], True )
          self._model_file_name = modelFileName

          input_shape = ( nFreqBands, blk_size, blk_size, 1 )
          input_layer = Input(input_shape, batch_size=batch_sz )
          print (input_layer)
          encoding_conv_1 = layers.Conv3D(filters=nbInputPatterns,
                                          kernel_size=(nFreqBands-4, 3, 3),
                                          strides=(1,1,1),
                                          padding='same',
                                          activation=active_function,
                                          use_bias = True,
                                          groups=1,
                                          data_format="channels_last")(input_layer)

          encoding_dropout = layers.Dropout(rate=drop_rate, seed = drop_seed )(encoding_conv_1)

          encoding_conv_2 = layers.Conv3D(filters=nbInputPatterns,
                                          kernel_size =(nFreqBands-8, 1, 1),
                                          strides=(1,1,1),
                                          padding='same',
                                          activation=active_function,
                                          use_bias=True,
                                          data_format="channels_last",
                                          groups = 1)(encoding_dropout)

#          code_layer = layers.Dense( encoded_dim, activation=active_function, use_bias=True)(encoding_conv_2)

          decoding_conv_1 = layers.Conv3D(filters=nbInputPatterns,
                                          kernel_size=(nFreqBands-8, 1, 1),
                                          strides=(1,1,1),
                                          padding='same',
                                          activation=active_function,
                                          use_bias = True,
                                          data_format="channels_last",
                                          #groups=1 )(code_layer)
                                          groups=1)(encoding_conv_2)

          decoding_dropout = layers.Dropout(rate=drop_rate, seed = drop_seed )(decoding_conv_1)

          decoding_conv_2 = layers.Conv3D(filters=nbInputPatterns,
                                          kernel_size=(nFreqBands-4, 3, 3),
                                          strides=(1,1,1),
                                          padding='same',
                                          activation=active_function,
                                          use_bias = True,
                                          data_format="channels_last",
                                          groups=1)(decoding_dropout)

          self._model = Model(input_layer, decoding_conv_2)
          self._model.summary()
          self._model.compile(optimizer='Adam', loss='binary_crossentropy')

     def train(self, input_train, input_test, batch_size, epochs ):
          print ( 'Input train data set: ', input_train.shape)
          print ( 'Input test data set: ', input_test.shape )
          print ( 'Model input shape: ', self._model.input_shape )
          loss = self._model.fit(input_train,
                                 input_test,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=True)

          if self._model_file_name is not None:
               self._model.save(self._model_file_name, save_format='h5')
          return loss

     def getDecodedImage ( self,encoded_imgs, batch_sz ):
          decoded_image = self._model.predict(batch_size=batch_sz, x=encoded_imgs)
          return decoded_image

