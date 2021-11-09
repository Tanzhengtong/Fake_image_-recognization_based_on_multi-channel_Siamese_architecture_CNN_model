import os
import random
import shutil
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dense
from spp.SpatialPyramidPooling import SpatialPyramidPooling
from utils.load_images import load_org_image, choice_predict, load_votes, load_meta, fix_index, dataAugmentation, \
    handle_image_pair1, clearDir
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.utils import to_categorical
from utils.model_utils import save_model, load_model

K.set_image_dim_ordering('th')


# 生成测试数据 用于不存在图片匹配测试
def testdata_loader3(index, prob):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()
    used_ids = []
    results = []
    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if v['vote'] > prob:
            if uid not in used_ids:
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i
                })

    centers = []
    pyrds = []
    labels = []
    tmpTrue = [[], []]

    count = 0
    random_index = index
    for i in range(len(results)):
        data = results[i]
        index = data['index']
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
        count += 1
        if i == random_index:
            tmpTrue[1].append(rpath)
        else:
            tmpTrue[0].append(rpath)
    tmpTrue = np.array(tmpTrue)
    # 覆盖图片目录
    dirpath = '/'
    clearDir()
    # 生成相似图片
    imageNo = 0
    for i in range(len(tmpTrue[0])):
        image1 = cv2.imread(str(tmpTrue[0][i]), 0)
        image2 = cv2.imread(str(tmpTrue[1][0]), 0)
        cv2.imwrite("./img/traintemp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", image1)
        cv2.imwrite("./img/traintemp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", image2)
        imageNo += 1
    # 重新读取文件路径
    train_sampleA = []
    for i in range(imageNo):
        train_sampleA.append("./img/traintemp/tempA/img" + str(i) + ".jpg")
    # 重新读取文件路径
    train_sampleB = []
    for i in range(imageNo):
        train_sampleB.append("./img/traintemp/tempB/img" + str(i) + ".jpg")

    train_sampleTrue = np.array([train_sampleA, train_sampleB])

    for i in range(imageNo):
        lpath = train_sampleTrue[0][i]
        rpath = train_sampleTrue[1][i]
        handle_image_pair1(lpath, rpath, centers, pyrds)

    for i in range(imageNo):
        if i == random_index:
            labels.append(0)
        else:
            labels.append(0)
    print(len(centers), len(pyrds), len(labels))
    centers = np.array(centers)
    pyrds = np.array(pyrds)
    return centers, pyrds, labels


# 生成数据集 尺寸不一致 用于预测
def testdata_loader1(prob=0.5):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()

    used_ids = []
    results = []

    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if v['vote'] > prob:
            if uid not in used_ids:
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i
                })

    centers = []
    pyrds = []
    labels = []
    tmpTrue = [[], []]

    count = 0
    np.random.shuffle(results)

    for i in range(len(results)):
        data = results[i]
        index = data['index']
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
        count += 1
        tmpTrue[1].append(rpath)
        tmpTrue[0].append(rpath)

    tmpTrue = np.array(tmpTrue)

    # 覆盖图片目录
    dirpath = '/'
    clearDir()
    # 生成相同图片
    imageNo = 0
    for i in range(len(tmpTrue[0])):
        image1 = cv2.imread(str(tmpTrue[0][i]), 0)
        image2 = cv2.imread(str(tmpTrue[1][i]), 0)
        cv2.imwrite("./img/traintemp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", image1)
        cv2.imwrite("./img/traintemp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", image2)
        imageNo += 1

    print("Num of images is ", imageNo)
    # 重新读取文件路径
    train_sampleA = []
    for i in range(imageNo):
        train_sampleA.append("./img/traintemp/tempA/img" + str(i) + ".jpg")
    # 重新读取文件路径
    train_sampleB = []
    for i in range(imageNo):
        train_sampleB.append("./img/traintemp/tempB/img" + str(i) + ".jpg")

    print("After dataAugmentation: Train_sample length is ", len(os.listdir("./img/traintemp/tempA" + dirpath)))

    train_sampleTrue = np.array([train_sampleA, train_sampleB])
    train_sampleFalse = np.array([train_sampleA, train_sampleB])

    # 自制错误标签数据集
    np.random.shuffle(train_sampleFalse[0])
    np.random.shuffle(train_sampleFalse[1])
    # 检查是否匹配到了相同的图像
    length_train_sample = len(train_sampleFalse[0])
    for i in range(length_train_sample):
        left = int(train_sampleFalse[0][i].replace("./img/traintemp/tempA/img", "").replace(".jpg", ""))
        right = int(train_sampleFalse[1][i].replace("./img/traintemp/tempB/img", "").replace(".jpg", ""))
    if left / 6 == right / 6:
        index = left - 10 if left > 10 else left + 10
        train_sampleFalse[1][i] = train_sampleTrue[1][index]

    for i in range(imageNo):
        lpath = train_sampleTrue[0][i]
        rpath = train_sampleTrue[1][i]
        handle_image_pair1(lpath, rpath, centers, pyrds)
        labels.append(1)

    for i in range(imageNo):
        lpath = train_sampleFalse[0][i]
        rpath = train_sampleFalse[1][i]
        handle_image_pair1(lpath, rpath, centers, pyrds)
        labels.append(0)

    centers = np.array(centers)
    pyrds = np.array(pyrds)
    print(len(centers), len(pyrds), len(labels))

    # 打乱验证数据集
    np.random.seed(116)
    np.random.shuffle(centers)
    np.random.seed(116)
    np.random.shuffle(pyrds)
    np.random.seed(116)
    np.random.shuffle(labels)
    return centers, pyrds, labels


# 样本相似度匹配 用于存在图片匹配
def testdata_loader2(index, prob):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()
    used_ids = []
    results = []
    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if v['vote'] > prob:
            if uid not in used_ids:
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i
                })

    centers = []
    pyrds = []
    labels = []
    tmpTrue = [[], []]

    count = 0
    random_index = index
    for i in range(len(results)):
        data = results[i]
        index = data['index']
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
        count += 1
        if i == random_index:
            tmpTrue[0].append(rpath)
            tmpTrue[1].append(rpath)
        else:
            tmpTrue[0].append(rpath)
    tmpTrue = np.array(tmpTrue)
    # 覆盖图片目录
    dirpath = '/'
    clearDir()
    # 生成相似图片
    imageNo = 0
    for i in range(len(tmpTrue[0])):
        image1 = cv2.imread(str(tmpTrue[0][i]), 0)
        image2 = cv2.imread(str(tmpTrue[1][0]), 0)
        cv2.imwrite("./img/traintemp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", image1)
        cv2.imwrite("./img/traintemp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", image2)
        imageNo += 1
    # 重新读取文件路径
    train_sampleA = []
    for i in range(imageNo):
        train_sampleA.append("./img/traintemp/tempA/img" + str(i) + ".jpg")
    # 重新读取文件路径
    train_sampleB = []
    for i in range(imageNo):
        train_sampleB.append("./img/traintemp/tempB/img" + str(i) + ".jpg")

    train_sampleTrue = np.array([train_sampleA, train_sampleB])

    for i in range(imageNo):
        lpath = train_sampleTrue[0][i]
        rpath = train_sampleTrue[1][i]
        handle_image_pair1(lpath, rpath, centers, pyrds)

    for i in range(imageNo):
        if i == random_index:
            labels.append(1)
        else:
            labels.append(0)
    print(len(centers), len(pyrds), len(labels))
    centers = np.array(centers)
    pyrds = np.array(pyrds)
    return centers, pyrds, labels


def testdata_loader4(prob):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()
    used_ids = []
    results = []
    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if v['vote'] > prob:
            if uid not in used_ids:
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i
                })
    tmpImage = []

    count = 0

    for i in range(len(results)):
        data = results[i]
        index = data['index']
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
        count += 1
        tmpImage.append(rpath)
    tmpImage = np.array(tmpImage)

    # 覆盖图片目录
    dirpath = '/'
    clearDir()
    # 生成相同图片
    imageNo = 0
    for i in range(len(tmpImage)):
        image1 = cv2.imread(str(tmpImage[i]), 0)
        cv2.imwrite("./img/traintemp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", image1)
        imageNo += 1
    print("Num of images is ", imageNo)
    # 重新读取文件路径
    train_sample = []
    for i in range(imageNo):
        train_sample.append("./img/traintemp/tempA/img" + str(i) + ".jpg")

    print("Train_sample length is ", len(os.listdir("./img/traintemp/tempA" + dirpath)))
    return train_sample


def testImagePair(indexl,indexr, lpath, rpath):
    start=time.clock()
    datas1, datas2, labels = [], [], []
    handle_image_pair1(lpath, rpath, datas1, datas2)

    if indexl==indexr:
        labels.append(1)
    else:
        labels.append(0)
    end=(time.clock()-start)
    print("each image takes :",end," seconds")
    return datas1,datas2,labels
# def targetImageExtract(index):

def predict():
    print("[INFO] Model Prediction...")
    start = time.clock()
    model = load_model('492a5e66-3537-11eb-8f29-69e1bbef6925',
                       custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    datas1, datas2, labels = testdata_loader1(0.5)
    results = model.predict([datas1, datas2])
    # results是矩阵 返回相似的概率 范围在0-1之间
    # 1 正确率
    y_pred = []
    for i in results:
        y_pred.append(1) if i > 0.5 else y_pred.append(0)
    compare = np.array(y_pred) == np.array(labels)
    count = np.sum(compare != 0)
    print("Model Accuracy is: ", count / len(labels))
    # 2 假阳性 假阴性
    fp = 0
    fn = 0
    for i in range(len(labels)):
        # 本来不相似但是识别为相似
        if (results[i] > 0.5 and labels[i] == 0):
            fp += 1
        # 本来相似但是识别为不相似
        if (results[i] < 0.5 and labels[i] == 1):
            fn += 1
    print("False positive is: ", fp)
    print("False negative is: ", fn)
    elapsed = (time.clock() - start)
    print(elapsed)


def test_exist():
    print("[INFO] Model Simulation Testing...")
    start = time.clock()
    model = load_model('492a5e66-3537-11eb-8f29-69e1bbef6925',
                       custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    acc, count95, count80, count50 = 0, 0, 0, 0
    for x in range(200, 210, 2):
        datas1, datas2, labels = testdata_loader2(x, 0.8)
        results = model.predict([datas1, datas2])
        # 大于0.95的匹配对
        count1 = np.sum(results > 0.95)
        count95 += count1
        # 大于0.8的匹配对
        count2 = np.sum(results > 0.8)
        count80 += count2
        # 大于0.5的匹配对个数
        count3 = np.sum(results > 0.5)
        count50 += count3
        print("大于0.95的个数有:", count1, "大于0.8的个数有:", count1, "大于0.5相似度的个数有:", count3)
        # 最大相似对
        maxIndex = np.argmax(results)
        groundTruth = np.argmax(labels)
        if maxIndex != groundTruth:
            image1 = datas1[maxIndex, 0, :, :]
            image2 = datas1[maxIndex, 3, :, :]
            plt.subplot(1, 2, 1)
            plt.imshow(image1)
            plt.text(0, 0, s=labels[maxIndex])
            plt.text(-20, -1, maxIndex)
            plt.subplot(1, 2, 2)
            plt.imshow(image2)
            plt.text(0, 0, s=results[maxIndex])
            plt.show()

            image1 = datas1[groundTruth, 0, :, :]
            image2 = datas1[groundTruth, 3, :, :]
            plt.subplot(1, 2, 1)
            plt.imshow(image1)
            plt.text(0, 0, s=labels[groundTruth])
            plt.text(-20, -1, groundTruth)
            plt.subplot(1, 2, 2)
            plt.imshow(image2)
            plt.text(0, 0, s=results[groundTruth])
            plt.show()
        else:
            image1 = datas1[maxIndex, 0, :, :]
            image2 = datas1[maxIndex, 3, :, :]
            plt.subplot(1, 2, 1)
            plt.imshow(image1)
            plt.text(0, 0, s=labels[maxIndex])
            plt.text(-20, -1, maxIndex)
            plt.subplot(1, 2, 2)
            plt.imshow(image2)
            plt.text(0, 0, s=results[maxIndex])
            plt.show()

            image1 = datas1[groundTruth, 0, :, :]
            image2 = datas1[groundTruth, 3, :, :]
            plt.subplot(1, 2, 1)
            plt.imshow(image1)
            plt.text(0, 0, s=labels[groundTruth])
            plt.text(-20, -1, groundTruth)
            plt.subplot(1, 2, 2)
            plt.imshow(image2)
            plt.text(0, 0, s=results[groundTruth])
            plt.show()
            acc += 1

    print("匹配度: ", acc)
    print("平均大于0.95的个数: ", count95 / 50)
    print("平均大于0.5的个数: ", count50 / 50)
    elapsed = (time.clock() - start)
    print(elapsed)


def test_notexist():
    print("[INFO] Model Simulation Testing...")
    start = time.clock()
    model = load_model('492a5e66-3537-11eb-8f29-69e1bbef6925',
                       custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    acc, count95, count80, count50 = 0, 0, 0, 0
    for x in range(200, 300, 2):
        datas1, datas2, labels = testdata_loader3(x, 0.8)
        results = model.predict([datas1, datas2])
        # 大于0.95的匹配对
        count1 = np.sum(results > 0.95)
        count95 += count1
        # 大于0.8的匹配对
        count2 = np.sum(results > 0.8)
        count80 += count2
        # 大于0.5的匹配对个数
        count3 = np.sum(results > 0.5)
        count50 += count3
        print("大于0.95的个数有:", count1, "大于0.8的个数有:", count1, "大于0.5相似度的个数有:", count3)
        # 最大相似对
        maxIndex = np.argmax(results)
        maxValue = np.max(results)
        if maxValue > 0.5:
            image1 = datas1[maxIndex, 0, :, :]
            image2 = datas1[maxIndex, 3, :, :]
            plt.subplot(1, 2, 1)
            plt.imshow(image1)
            plt.text(0, 0, s=labels[maxIndex])
            plt.text(-20, -1, maxIndex)
            plt.subplot(1, 2, 2)
            plt.imshow(image2)
            plt.text(0, 0, s=results[maxIndex])
            plt.show()
        else:
            acc += 1

    print("匹配度: ", acc)
    print("平均大于0.95的个数: ", count95 / 50)
    print("平均大于0.5的个数: ", count50 / 50)
    elapsed = (time.clock() - start)
    print(elapsed)


def test_singleimage():
    print("[INFO] Model Simulation Testing...")
    start = time.clock()
    model = load_model('492a5e66-3537-11eb-8f29-69e1bbef6925',
                       custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    elapsed = (time.clock() - start)
    print("Model Loading Time: ",elapsed)
    start=time.clock()
    exsitImage = testdata_loader4(0.5)
    elapsed=(time.clock()-start)
    print("Index Read Time: ",elapsed)
    start=time.clock()
    for x in range(0,100):
        lpath=exsitImage[x]
        for y in range(len(exsitImage)-1,-1,-1):
            rpath=exsitImage[y]
            datas1, datas2, labels =testImagePair(x,y,lpath,rpath)
            results = model.predict([datas1, datas2])
            if(results>0.5 and x==y):
                print("MATCHED",results)
                break
            if(y==len(exsitImage)):
                print("MISSED!")
    elapsed=(time.clock()-start)
    print("Prediction Time :",elapsed)

def test_batchimage():
    print("[INFO] Model Simulation Testing...")
    start = time.clock()
    model = load_model('492a5e66-3537-11eb-8f29-69e1bbef6925',
                       custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    elapsed = (time.clock() - start)
    print("Model Loading Time: ",elapsed)
    start=time.clock()
    # existImage 保存已有照片的路径信息
    exsitImage = testdata_loader4(0.5)

    elapsed=(time.clock()-start)
    print("Index Read Time: ",elapsed)
    start=time.clock()
    for x in range(0,2):
        lpath=exsitImage[x]
        for y in range(len(exsitImage)-1,-1,-1):
            rpath=exsitImage[y]
            datas1, datas2, labels =testImagePair(x,y,lpath,rpath)
            results = model.predict([datas1, datas2])
            if(results>0.5 and x==y):
                print("MATCHED ",results)
                break
            if(y==0):
                print("MISSED!")
    elapsed=(time.clock()-start)
    print("Prediction Time :",elapsed)



# predict()
# test_exist()
# test_notexist()
# test_singleimage()
test_batchimage()

