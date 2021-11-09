import time

import keras
from keras import backend as K
from models.model import Model
import sys
from utils.utils import get_option
from utils.model_utils import save_model, load_model
import multiprocessing
from utils.load_images import *
# from utils.load_images import load_image, load_image_new,load_image_dif_size,image_archive
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from net.channel_net import ChannelNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from PIL import Image

sys.setrecursionlimit(1000000)
np.random.seed(1337)  # For Reproducibility

EPOCHS = 20
INIT_LR = 1e-3
BS = 16  # 32 #20  # 128


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


class ImageSimilarityModel(Model):

    def __init__(self, uid, config={}):
        super(ImageSimilarityModel, self).__init__(uid)
        self.config = config
        self.load_config()

        print(self.get_config_dict())

    def load_config(self):
        self.n_iterations = int(get_option(self.config, 'n_iterations', 1))  # ideally more..
        self.n_exposures = int(get_option(self.config, 'n_exposures', 10))
        self.batch_size = int(get_option(self.config, 'batch_size', 64))
        self.n_epoch = int(get_option(self.config, 'n_epoch', 10))
        self.lr = float(get_option(self.config, 'learn_rate', 0.0005))
        self.cpu_count = int(get_option(self.config, 'cpu_count', multiprocessing.cpu_count()))
        self.test_size = float(get_option(self.config, 'test_size', 0.1))
        self.validation_split = float(get_option(self.config, 'validation_split', 0.2))
        self.model_dir = str(get_option(self.config, 'model_dir', '/mnt/mydata/deep_siren_models'))
        self.data_dir = str(get_option(self.config, 'data_dir', '/mnt/mydata/news/tdata_all_xls'))
        self.retrain_flag = bool(get_option(self.config, 'retrain_flag', False))

        pass

    def get_config_dict(self):
        config = {}
        config['n_iterations'] = self.n_iterations
        config['n_exposures'] = self.n_exposures
        config['batch_size'] = self.batch_size
        config['n_epoch'] = self.n_epoch
        config['cpu_count'] = self.cpu_count
        config['test_size'] = self.test_size
        config['validation_split'] = self.validation_split
        config['model_dir'] = self.model_dir
        config['data_dir'] = self.data_dir
        config['retrain_flag'] = self.retrain_flag
        config['learn_rate'] = self.lr

        return config

    def description(self):

        return 'image similarity net,use conv2d'

    def get_data(self):
        num_classes = 0
        datas = []
        labels = []
        copy = 200
        for f in os.listdir(self.data_dir):
            num_classes += 1
            path = os.path.join(self.data_dir, f)
            print(path)
            image = cv2.imread(path)
            print(image)
            image = cv2.resize(image, (66, 66))

            image = img_to_array(image)
            t = int(f[4])
            for i in range(copy):
                datas.append(image)
                labels.append(t)

        datas = np.array(datas, dtype="float") / 255.0
        labels = np.array(labels)

        return datas, labels, num_classes

    def split(self, datas1, datas2, labels, testsize=0.1, valsize=0.1):
        # train 训练集
        # val 验证集 训练时验证
        # test 测试集 不参与训练
        trainX1 = []
        testX1 = []
        trainX2 = []
        testX2 = []
        trainY = []
        testY = []
        valX1 = []
        valX2 = []
        valY = []
        # train 和 val共同占用的比重
        length = int(len(datas1) * (1 - testsize))
        # val占用 train的比重
        testlength = int(length * valsize)
        for i in range(length):
            if (i < testlength):
                valX1.append(datas1[i])
                valX2.append(datas2[i])
                valY.append(labels[i])
            else:
                trainX1.append(datas1[i])
                trainX2.append(datas2[i])
                trainY.append(labels[i])
        for j in range(length, len(datas1)):
            testX1.append(datas1[j])
            testX2.append(datas2[j])
            testY.append(labels[j])
        print("train: ", len(trainX1), "test: ", len(testX1), "val: ", len(valX1))

        return trainX1, testX1, valX1, trainX2, testX2, valX2, trainY, testY, valY

    def load_train(self, x_train, y_train, x_test, y_test):
        print('loading model......')
        model = load_model(self.uid, self.model_dir)
        print('Compiling the Model...')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])

        print("PreEvaluate...")
        score = model.evaluate(x_test, y_test,
                               batch_size=self.batch_size)
        print('PreScore', score)

        print("Train...")
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)
        model.fit(x_train, y_train, batch_size=self.batch_size,
                  epochs=self.n_epoch, verbose=1, validation_split=self.validation_split)

        print("Evaluate...")
        score = model.evaluate(x_test, y_test,
                               batch_size=self.batch_size)

        save_model(model, self.uid, self.model_dir)
        print('Test score:', score)

    def train(self):
        image_size, archive_index = image_archive(0.3)
        print("[INFO] Compiling Model...")
        model = ChannelNet.build()
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        print("[INFO] Training Network...")
        H = []
        for i in range(len(image_size)):
            print("[INFO] Current processing image size is:", image_size[i])
            # 读取数据
            datas1, datas2, labels = load_image_dif_size(image_size[i], archive_index[i], 0.0)
            # 训练集 测试集 验证集 分割
            trainX1, testX1, valX1, trainX2, testX2, valX2, trainY, testY, valY = self.split(datas1, datas2, labels,
                                                                                             0.1, 0.4)
            # 模型训练
            H = model.fit([trainX1, trainX2], trainY, batch_size=BS, validation_data=([valX1, valX2], valY),
                          epochs=EPOCHS, verbose=2, shuffle=True)

        print("[INFO] Testing Model...")
        score = model.evaluate([testX1, testX2], testY, batch_size=BS)
        print(score)
        print("[INFO] Saving Model...")
        save_model(model, self.uid)
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('./plot.jpg')
        plt.show()

        print("[INFO] Completed...")

    def predict(self, param):
        print('Loading model...')
        model = load_model(self.uid, self.model_dir)

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])


class ImageSimilarityModelInstance:

    def __init__(self, uid, config):
        self.config = config
        self.uid = uid
        self.model_dir = str(get_option(self.config, 'model_dir', '/mnt/mydata/deep_siren_models'))
        self.tmp_dir = str(get_option(self.config, 'tmp_dir', '/mnt/mydata/deep_siren_models/tmp_dir'))
        self.model = load_model(self.uid, self.model_dir)
        self.tmp_dir = './'

    def iter_frames(self, im):
        try:
            i = 0
            while 1:
                im.seek(i)
                imframe = im.copy()
                print(i)
                if i == 0:
                    palette = imframe.getpalette()
                else:
                    imframe.putpalette(palette)
                yield imframe
                i += 1
        except Exception as e:
            print(e)
            pass

    def trans(self, inpath, outpath):
        im = Image.open(inpath)
        for i, frame in enumerate(self.iter_frames(im)):
            frame.save(outpath, **frame.info)
            break

    def predict(self, param):
        if (not os.path.exists(self.tmp_dir)):
            os.makedirs(self.tmp_dir)
        path = None
        type = param.get('body_type')
        body = param.get('body')
        if (type and body):
            cur_path = os.path.join(self.tmp_dir, 'tmp.' + type)
            with open(cur_path, 'wb') as wf:
                wf.write(body)

            if (type == 'gif'):
                inpath = cur_path
                outpath = os.path.join(self.tmp_dir, 'tmp.png')

                self.trans(inpath, outpath)

                os.remove(inpath)
                path = outpath
            else:
                path = cur_path

        if (path is None):
            raise Exception('no inpu data')
        else:
            image = cv2.imread(path)
            image = cv2.resize(image, (66, 66))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            res = self.model.predict(image)[0]
            res = int(np.argmax(res, axis=0))
            return res

        raise Exception('unkown')


'''def train(self):
    print("[INFO] Compiling Model...")
    model = ChannelNet.build()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    print("[INFO] Training Network...")
    datas1, datas2, labels = load_image_new(200,200)
    trainX1, testX1, valX1, trainX2, testX2, valX2, trainY, testY, valY = self.split(datas1, datas2, labels,0.2)
    model.fit([trainX1, trainX2], trainY, batch_size=BS, validation_data=([testX1, testX2], testY),
              epochs=EPOCHS, verbose=2, shuffle=True)

    print("[INFO] Testing Model...")
    score = model.evaluate([valX2, valX1], valY, batch_size=BS)
    print(score)

    print("[INFO] Saving Model...")
    save_model(model, self.uid)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), model.history["acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), model.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('./plot.jpg')
    plt.show()

    print("[INFO] Completed...")'''

"""def train(self):
    print("[INFO] Compiling Model...")
    model = ChannelNet.build()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    print("[INFO] Training Network...")
    datas1, datas2, labels = load_image_same_size(200)
    trainX1, testX1, valX1, trainX2, testX2, valX2, trainY, testY, valY = self.split(datas1, datas2, labels,0.2)
    H=model.fit([trainX1, trainX2], trainY, batch_size=BS, validation_data=([testX1, testX2], testY),
                epochs=EPOCHS, verbose=2, shuffle=True)

    print("[INFO] Testing Model...")
    score = model.evaluate([valX2, valX1], valY, batch_size=BS)
    print(score)

    print("[INFO] Saving Model...")
    save_model(model, self.uid)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('./plot.jpg')
    plt.show()

    print("[INFO] Completed...")"""
'''
       #cross-validation
       k=4
       num_val_samples=(len(datas1)-1000)//k
       print(len(datas1))
       print(len(datas1)//k)
       all_scores=[]
       for i in range(k):
           print("processing fold  # ",i)
           val_data1=datas1[i*num_val_samples:(i+1)*num_val_samples]
           val_data2=datas2[i*num_val_samples:(i+1)*num_val_samples]
           val_targets=labels[i*num_val_samples:(i+1)*num_val_samples]
           train_data1=np.concatenate([datas1[:i*num_val_samples+1],datas1[(i+1)*num_val_samples:]],axis=0)
           train_data2=np.concatenate([datas2[:i*num_val_samples+1],datas2[(i+1)*num_val_samples:]],axis=0)
           train_target=np.concatenate([labels[:i*num_val_samples+1],labels[(i+1)*num_val_samples:]],axis=0)
           test_data1=datas1[:1000]
           test_data2=datas2[:1000]
           test_targets=labels[:1000]

           print("[INFO] Compiling Model...")
           model = ChannelNet.build()
           model.summary()
           model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
           model.fit([train_data1,train_data2], train_target, epochs=10, batch_size=20, shuffle=True, verbose=2, validation_data=([val_data1,val_data2],val_targets))

           score = model.evaluate([test_data2,test_data1], test_targets,batch_size=20)
           print(score)
           all_scores.append(score)
       print(all_scores)
       '''
