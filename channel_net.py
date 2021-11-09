from keras.models import Model, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from spp.SpatialPyramidPooling import SpatialPyramidPooling
import keras

K.set_image_dim_ordering('th')


class ChannelNet:
    @staticmethod
    # 论文模型
    def build():
        input_shape = (6, None, None)
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        # intializer definition
        initialize_weights = keras.initializers.he_normal(seed=None)
        initialize_bias = 'zeros'
        # build convnet to use in each siamese 'leg'
        convnet = keras.models.Sequential()
        # 64 128 256 4096
        # 卷积层
        convnet.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                        input_shape=input_shape,
                                        kernel_initializer=initialize_weights))
        # 最大池化层
        convnet.add(keras.layers.MaxPooling2D())
        # 卷积层
        convnet.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
                                        kernel_initializer=initialize_weights,
                                        bias_initializer=initialize_bias))
        # 最大池化层
        convnet.add(keras.layers.MaxPooling2D())
        # 卷积层
        convnet.add(keras.layers.Conv2D(128, (3, 3), activation='relu',
                                        kernel_initializer=initialize_weights,
                                        bias_initializer=initialize_bias))
        # # 最大池化层
        # convnet.add(keras.layers.MaxPooling2D())
        # convnet.add(keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer=initialize_weights,
        #                                 bias_initializer=initialize_bias))

        # spp层
        convnet.add(SpatialPyramidPooling([1, 2, 4]))
        # Input: the input Tensor to invert ([batch, channel, width, height])
        # 全连接层
        convnet.add(keras.layers.Dense(512, activation="sigmoid",
                                       kernel_initializer=initialize_weights,
                                       bias_initializer=initialize_bias))
        # convnet.add(keras.layers.Dropout(0.5))
        # call the convnet Sequential model on each of the input tensors so params will be shared
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)
        # layer to merge two encoded inputs with the l1 distance between them
        # L1_layer = keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_layer = keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])
        # print("L1_distance is", L1_distance)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        siamese_net.summary()
        return siamese_net

    '''
    def build():
        # 配合原load_image
        input_shape = (2, 100, 100)
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        print("left_input is ", left_input)
        # intializer definition
        initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.51, seed=50001)
        initialize_bias = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1221)
        # build convnet to use in each siamese 'leg'
        convnet = keras.models.Sequential()
        convnet.add(keras.layers.Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                                        kernel_initializer=initialize_weights,
                                        kernel_regularizer=keras.regularizers.l2(2e-4)))
        convnet.add(keras.layers.MaxPooling2D())
        convnet.add(keras.layers.Conv2D(128, (7, 7), activation='relu',
                                        kernel_regularizer=keras.regularizers.l2(2e-4),
                                        kernel_initializer=initialize_weights,
                                        bias_initializer=initialize_bias))
        convnet.add(keras.layers.MaxPooling2D())
        convnet.add(keras.layers.Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                                        kernel_regularizer=keras.regularizers.l2(2e-4),
                                        bias_initializer=initialize_bias))
        convnet.add(keras.layers.MaxPooling2D())
        convnet.add(keras.layers.Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                                        kernel_regularizer=keras.regularizers.l2(2e-4),
                                        bias_initializer=initialize_bias))
        convnet.add(keras.layers.Flatten())
        convnet.add(keras.layers.Dense(4096, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(1e-3),
                                       kernel_initializer=initialize_weights,
                                       bias_initializer=initialize_bias))
        # convnet.add(keras.layers.Dropout(rate=0.05))
        # call the convnet Sequential model on each of the input tensors so params will be shared
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)
        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
        print("prediction shape is", prediction)
        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        return siamese_net
        '''

    '''
    # 配合原load_image
    input_shape = (2, 100, 100)
    # 配合新load_image_new
    # input_shape = (2, 200, 200)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    print("left_input is ", left_input)
    # intializer definition
    initialize_weights = keras.initializers.RandomNormal(mean=0.0, stddev=0.51, seed=50001)
    initialize_bias = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1221)
    # build convnet to use in each siamese 'leg'
    convnet = keras.models.Sequential()
    convnet.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape,
                                    kernel_initializer=initialize_weights,
                                    kernel_regularizer=keras.regularizers.l2(2e-4)))
    convnet.add(keras.layers.MaxPooling2D())
    convnet.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(2e-4),
                                    kernel_initializer=initialize_weights,
                                    bias_initializer=initialize_bias))
    convnet.add(keras.layers.MaxPooling2D())
    convnet.add(keras.layers.Conv2D(100, (1, 1), activation='relu', kernel_initializer=initialize_weights,
                                    kernel_regularizer=keras.regularizers.l2(2e-4),
                                    bias_initializer=initialize_bias))
    # convnet.add(keras.layers.MaxPooling2D())
    # convnet.add(keras.layers.Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
    #                                 kernel_regularizer=keras.regularizers.l2(2e-4),
    #                                 bias_initializer=initialize_bias))
    convnet.add(keras.layers.Flatten())
    convnet.add(keras.layers.Dense(1024, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(1e-3),
                                   kernel_initializer=initialize_weights,
                                   bias_initializer=initialize_bias))
    # convnet.add(keras.layers.Dropout(rate=0.05))
    # call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    # layer to merge two encoded inputs with the l1 distance between them
    L1_layer = keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
    print("prediction shape is", prediction)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net
    '''

    '''
    test1=1
    inputShape=(2,None,None)
    input1 = Input(shape=inputShape)
    model1 = Convolution2D(32,(3,3),padding="same",input_shape=inputShape,data_format='channels_first',activation='relu')(input1)
    print(model1.shape)
    model1 = Convolution2D(32,(3,3),padding="same",activation='relu')(model1)
    print(model1.shape)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    print(model1.shape)
    model1 = Convolution2D(64,(3,3),padding="same",activation='relu')(model1)
    print(model1.shape)
    model1 = Convolution2D(64,(3,3),padding="same",activation='relu')(model1)
    print(model1.shape)
    model1 = SpatialPyramidPooling([1,2,4])(model1)

    input2 = Input(shape=inputShape)
    model2 = Convolution2D(32,(3,3),padding="same",input_shape=inputShape,data_format='channels_first',activation='relu')(input2)
    model2 = Convolution2D(32,(3,3),padding="same",activation='relu')(model2)
    print(model2.shape)
    model2 = MaxPooling2D(pool_size=(2, 2))(model2)
    print(model2.shape)
    model2 = Convolution2D(64,(3,3),padding="same",activation='relu')(model2)
    print(model2.shape)
    model2 = Convolution2D(64,(3,3),padding="same",activation='relu')(model2)
    print(model2.shape)
    model2 = SpatialPyramidPooling([1,2,4])(model2)
    
    print(model1.shape,model2.shape)
    out = K.concatenate([model1,model2])
    print(out.shape)
    out = Dense(units=768, activation='sigmoid', use_bias=True,
                             kernel_initializer=keras.initializers.he_normal(seed=None),
                             bias_initializer='zeros')(out)
    out = Dense(2,activation='softmax')(out)
    
    model = Model(inputs=[input1,input2],outputs=[out])

    # return the constructed network architecture
    return model
    '''
