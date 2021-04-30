import tensorflow as tf


class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)


    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
        shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)


    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)



class AutoEncoder(tf.keras.Model):

  def __init__(self, inputs_width):
    super(AutoEncoder, self).__init__()
    self.__dense11 = tf.keras.layers.Dense(inputs_width, activation="selu")
    self.__dense12 = tf.keras.layers.Dense(inputs_width//2, activation="selu")
    self.__reg = tf.keras.layers.ActivityRegularization(l1=1e-3)
    self.encoder = tf.keras.Sequential([
        self.__dense11,
        self.__dense12,
        self.__reg
    ])

    self.__dense21 = DenseTranspose(self.__dense12, activation='selu')
    self.__dense22 = DenseTranspose(self.__dense11, activation='linear')
    self.decoder = tf.keras.Sequential([
        self.__dense21,
        self.__dense22     
    ])

    
  def call(self, inputs):
    encoder = self.encoder(inputs)
    decoder = self.decoder(encoder)
    return decoder