import math
import random
import shutil
import matplotlib.pyplot as plt
import keras
import numpy
import pandas as pd
import pickle
# from spp.SpatialPyramidPooling import SpatialPyramidPooling
import csv
import os
import cv2
import numpy as np

'''
with open('/mnt/mydata/DLData/Totally-Looks-Like-Data/lists_with_human_queries.pkl','rb') as rf:
    query_data = pickle.load(rf)
    print(type(query_data))
    print(query_data.keys())
    print(query_data['no_faces'].keys())
    print(query_data['no_faces']['random'].keys())

'''


# # 双通道 灰度图 中心采样和中心采样配对
# def load_image_new(width_res, length_res):
#     """
#     :param per_limit: 读取照片数量的限制
#     :param width_res: 照片裁剪的宽度
#     :param length_res: 照片裁剪的长度
#     :return:
#     """
#     votes = load_votes()
#     index_images, no_faces, no_dups = load_meta()
#
#     used_ids = []
#     results = []
#
#     for i, image in index_images:
#         uid = int(image.replace('.jpg', ''))
#         v = votes.get(uid, None)
#         if (v['vote'] > 0.95):
#             if uid not in used_ids:
#                 used_ids.append(uid)
#                 results.append({
#                     'uid': uid,
#                     'index': i
#                 })
#
#     centers = []
#     pyrds = []
#     labels = []
#     tmpTrue = [[], []]
#     tmpFalse = [[], []]
#
#     count = 0
#     np.random.shuffle(results)
#
#     for i in range(len(results)):
#         data = results[i]
#         uid = data['uid']
#         index = data['index']
#
#         lpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
#         rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
#         count += 1
#         tmpTrue[0].append(lpath)
#         tmpTrue[0].append(rpath)
#
#     tmpTrue = np.array(tmpTrue)
#
#     # 覆盖图片目录
#     dirpath = '/'
#     del_list = ["./img/testtemp/tempA", "./img/testtemp/tempB", "./img/test/true", "./img/test/false",
#                 "./img/traintemp/tempA", "./img/traintemp/tempB", "./img/train/true", "./img/train/false"]
#     del_list = [x + dirpath for x in del_list]
#     print("del_list are ", del_list)
#     for f in del_list:
#         if not os.path.exists(f):
#             os.makedirs(f)
#         else:
#             shutil.rmtree(f)
#             os.makedirs(f)
#     start = 0
#     end = dataAugmentation("train", dirpath, start, len(tmpTrue[0]), tmpTrue[0])
#
#     # 重新读取文件路径
#     train_sampleA = []
#     for i in range(start, end):
#         train_sampleA.append("./img/traintemp/tempA/img" + str(i) + ".jpg")
#     # 重新读取文件路径
#     train_sampleB = []
#     for i in range(start, end):
#         train_sampleB.append("./img/traintemp/tempB/img" + str(i) + ".jpg")
#
#     print("after re arrange: train_sampleB length is ", len(os.listdir("./img/traintemp/tempB" + dirpath)))
#     print("after re arrange: train_sampleA length is ", len(os.listdir("./img/traintemp/tempA" + dirpath)))
#
#     train_sampleTrue = np.array([train_sampleA, train_sampleB])
#     train_sampleFalse = np.array([train_sampleA, train_sampleB])
#
#     # 自制错误标签数据集
#     np.random.shuffle(train_sampleFalse[0])
#     np.random.shuffle(train_sampleFalse[1])
#     # 检查是否匹配到了相同的图像
#     length_train_sample = len(train_sampleFalse[0])
#     for i in range(length_train_sample):
#         left = int(train_sampleFalse[0][i].replace("./img/traintemp/tempA/img", "").replace(".jpg", ""))
#         right = int(train_sampleFalse[1][i].replace("./img/traintemp/tempB/img", "").replace(".jpg", ""))
#         if left / 6 == right / 6:
#             print("left is ", left, "right is ", right)
#             index = left - 6 if left > 6 else left + 6
#             print("index is", index, "right is", right)
#             train_sampleFalse[1][i] = train_sampleTrue[1][index]
#     # 中心采样 下采样
#     for i in range(end):
#         lpath = train_sampleTrue[0][i]
#         rpath = train_sampleTrue[1][i]
#         handle_image_pair(lpath, rpath, width_res, length_res, centers, pyrds)
#         labels.append(1)
#
#     for i in range(end):
#         lpath = train_sampleFalse[0][i]
#         rpath = train_sampleFalse[1][i]
#         handle_image_pair(lpath, rpath, width_res, length_res, centers, pyrds)
#         labels.append(0)
#
#     print(len(centers), len(pyrds), len(labels))
#
#     centers = np.array(centers)  # , dtype=object)
#     pyrds = np.array(pyrds)  # , dtype=object)
#     # labels = np.array(labels)
#     # print("data1 dim ", np.array(centers).ndim)
#     # print("data2 dim ", np.array(pyrds).ndim)
#     # print("labels dim ", np.array(labels).ndim)
#     print("shape are :")
#     print(centers.shape, pyrds.shape, labels.shape)
#     # 打乱验证数据集
#     np.random.seed(117)
#     np.random.shuffle(centers)
#     np.random.seed(117)
#     np.random.shuffle(pyrds)
#     np.random.seed(117)
#     np.random.shuffle(labels)
#
#     return centers, pyrds, labels


def dataExtract(probability):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()

    used_ids = []
    results = []
    tmpImage = []
    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if (v['vote'] > probability):
            if uid not in used_ids:
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i
                })

    for i in range(len(results)):
        data = results[i]
        index = data['index']
        lpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
        # rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
        tmpImage.append(lpath)
        # tmpImage.append(rpath)

    tmpImage = np.array(tmpImage)
    return tmpImage


# 双通道 灰度图 中心采样和中心采样配对 不限制尺寸大小
def load_image_dif_size(image_size, archive_index, prob):
    # extract specific image paths
    rawData = dataExtract(prob)
    tmpImage = rawData[archive_index]
    print("Current extracted image number is: ", len(tmpImage))
    centers = []
    pyrds = []
    labels = []
    # 根目录
    dirpath = '/'
    start = 0
    end = dataAugmentation("train", dirpath, start, len(tmpImage), tmpImage)
    # 重新读取文件路径
    train_sampleA = []
    for i in range(start, end):
        train_sampleA.append("./img/traintemp/tempA/img" + str(i) + ".jpg")
    train_sampleB = []
    for i in range(start, end):
        train_sampleB.append("./img/traintemp/tempB/img" + str(i) + ".jpg")
    print("After data Augmentation, the data length is ", len(os.listdir("./img/traintemp/tempB" + dirpath)))
    # 生成正确数据集
    train_sampleTrue = np.array([train_sampleA, train_sampleB])
    # 生成错误数据集
    train_sampleFalse = np.array([train_sampleA, train_sampleB])
    print(np.array(train_sampleTrue).shape)
    # 自制错误标签数据集
    # print(np.array(train_sampleFalse)[0,0:10])
    # print(np.array(train_sampleFalse)[1,0:10])
    np.random.shuffle(train_sampleFalse[0])
    np.random.shuffle(train_sampleFalse[1])
    # print(np.array(train_sampleFalse)[0,0:10])
    # print(np.array(train_sampleFalse)[1,0:10])
    # 检查是否匹配到了相同的图像
    length_train_sample = len(train_sampleFalse[0])
    for i in range(length_train_sample):
        left = int(train_sampleFalse[0][i].replace("./img/traintemp/tempA/img", "").replace(".jpg", ""))
        right = int(train_sampleFalse[1][i].replace("./img/traintemp/tempB/img", "").replace(".jpg", ""))
        # print(left, right)
        if left / 6 == right / 6:
            print("left is ", left, "right is ", right)
            index = left - 6 if left > 6 else left + 6
            print("index is", index, "right is", right)
            train_sampleFalse[1][i] = train_sampleTrue[1][index]
    # for i in range(length_train_sample):
    #     left = int(train_sampleFalse[0][i].replace("./img/traintemp/tempA/img", "").replace(".jpg", ""))
    #     right = int(train_sampleFalse[1][i].replace("./img/traintemp/tempB/img", "").replace(".jpg", ""))
    #     print(left, right, left/6==right/6)


    for i in range(end):
        lpath = train_sampleTrue[0][i]
        rpath = train_sampleTrue[1][i]
        handle_image_pair1(lpath, rpath, centers, pyrds)
        labels.append(1)

    for i in range(end):
        lpath = train_sampleFalse[0][i]
        rpath = train_sampleFalse[1][i]
        handle_image_pair1(lpath, rpath, centers, pyrds)
        labels.append(0)

    print(len(centers), len(pyrds), len(labels))

    centers = np.array(centers)
    pyrds = np.array(pyrds)
    print(centers.shape,pyrds.shape)
    # 打乱验证数据集
    np.random.seed(117)
    np.random.shuffle(centers)
    np.random.seed(117)
    np.random.shuffle(pyrds)
    np.random.seed(117)
    np.random.shuffle(labels)

    return centers, pyrds, labels

# 读取数据集路径
def image_archive(prob):
    tmpImage = dataExtract(prob)
    archiveIndex = []
    imageSize = []

    # 覆盖图片目录
    dirpath = '/'
    del_list = ["./img/testtemp/tempA", "./img/testtemp/tempB", "./img/test/true", "./img/test/false",
                "./img/traintemp/tempA", "./img/traintemp/tempB", "./img/train/true", "./img/train/false"]
    del_list = [x + dirpath for x in del_list]
    for f in del_list:
        if not os.path.exists(f):
            os.makedirs(f)
        else:
            shutil.rmtree(f)
            os.makedirs(f)

    for i in range(len(tmpImage)):
        image = cv2.imread(str(tmpImage[i]), 0)
        shape = image.shape
        # 如果未有该图片尺寸则新加
        if shape not in imageSize:
            imageSize.append(shape)
            archiveIndex.append([])
        # 添加下标
        archiveIndex[imageSize.index(shape)].append(i)
    return imageSize, archiveIndex


def load_votes():
    path = '/mnt/mydata/DLData/Totally-Looks-Like-Data/votes.csv'
    results = {}
    with open(path) as rf:
        res = csv.reader(rf)
        for line in res:
            if (line[0] == 'id'):
                continue

            uid = int(line[0])
            f = int(line[3])
            t = int(line[4])
            results[uid] = {
                'file': line[1],
                'vote': (t - f) / (t + f)
            }
    return results


def load_meta():
    with open('/mnt/mydata/DLData/Totally-Looks-Like-Data/metadata.pkl', 'rb') as rf:
        meta = pickle.load(rf)
        images = meta['image_list']
        no_faces = []
        no_dups = []

        for i in range(len(meta['no_faces'])):
            if (meta['no_faces'][i]):
                no_faces.append((i, images[i]))

        for i in range(len(meta['no_dups'])):
            if (meta['no_dups'][i]):
                no_dups.append((i, images[i]))

        index_images = []
        for i in range(len(images)):
            index_images.append((i, images[i]))
        return index_images, no_faces, no_dups

def fix_index(index):
    str_i = str(index)
    l = len(str_i)
    res = ''
    for i in range(5 - l):
        res += '0'
    res += str_i
    return res


def rotate90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img


def rotate180(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, -1)
    return new_img


def rotate270(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    return new_img


# 中心采样
def centeround(image):
    shape = image.shape
    b, w, h = shape
    # 灰度图二维尺寸提取
    # w,h=shape
    # startw=math.ceil(w * 0.25)
    startw = int(w * 0.25)
    endw = math.ceil(w * 0.75)
    starth = int(h * 0.25)
    endh = int(h * 0.75)
    newimage = image[:, startw:endw, starth:endh]
    # plt.imshow(np.transpose(newimage, (1, 2, 0)))
    # plt.show()
    # newimage = image[startw:endw, starth:endh]
    return newimage
# 三维图片下采样
def pyr_down(image):
    r,g,b=image
    r=cv2.pyrDown(r)
    g=cv2.pyrDown(g)
    b=cv2.pyrDown(b)
    newimage=np.array([r,g,b])
    # plt.imshow(np.transpose(newimage, (1,2,0)))
    # plt.show()
    return newimage
# 灰度图下采样
# def pyr_down(image):
#     newimage=cv2.pyrDown(image)
#     return newimage

# 图A的中心采样和下采样压缩在一起
def image_pair_to_arrays(image1, image2):
    centerimage1 = centeround(image1)
    centerimage2 = centeround(image2)
    pyrimage1 = pyr_down(image1)
    pyrimage2 = pyr_down(image2)
    # print("center shape is", np.array(centerimage1).shape)
    # print("pyrimage shape is ",np.array(pyrimage1).shape)
    # new_image1=np.array([centerimage1,centerimage2])
    # new_image1 = np.concatenate((centerimage1, pyrimage1), axis=0)
    new_image1 = np.concatenate((centerimage1, centerimage2), axis=0)
    new_image1 = np.array(new_image1)
    new_image1 = new_image1 / 255
    # print("left shape is ",np.array(new_image1).shape)
    # new_image2=np.array([pyrimage1,pyrimage2])
    # new_image2 = np.concatenate((centerimage2, pyrimage2), axis=0)
    new_image2 = np.concatenate((pyrimage1, pyrimage2), axis=0)
    new_image2 = np.array(new_image2)
    new_image2 = new_image2 / 255
    # print("right shape is ",np.array(new_image2).shape)
    return new_image1, new_image2

# 取消尺寸大小限制
def handle_image_pair1(lpath, rpath, datas1, datas2):
    #读取RGB图片
    limage = cv2.imread(lpath)
    rimage = cv2.imread(rpath)
    # 灰度图处理
    # limage = cv2.cvtColor(limage, cv2.COLOR_BGR2GRAY)
    # rimage = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
    # 归一化
    # limage = limage / 255
    # rimage = rimage / 255
    # 处理一维图片 增加batch维度
    # limage = np.expand_dims(limage, axis=0)
    # rimage = np.expand_dims(rimage, axis=0)
    # 调整矩阵维度顺序
    limage = np.transpose(limage, (2, 0, 1))
    rimage = np.transpose(rimage, (2, 0, 1))
    # data1, data2 = limage, rimage
    data1, data2 = image_pair_to_arrays(limage, rimage)
    datas1.append(data1)
    datas2.append(data2)

# 图A 图B压缩在一起进行下采样和中心采样
def handle_image_pair(lpath, rpath, width_res, length_res, centers=[], pyrds=[]):
    limage = cv2.imread(lpath)
    limage = cv2.cvtColor(limage, cv2.COLOR_BGR2GRAY)

    rimage = cv2.imread(rpath)
    rimage = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)

    limage = cv2.resize(limage, (width_res, length_res))
    rimage = cv2.resize(rimage, (width_res, length_res))

    if (pyrds is not None):
        center, pyrd = image_pair_to_arrays(limage, rimage)
        centers.append(center)
        pyrds.append(pyrd)


def choice_predict(pair=True):
    index_images, no_faces, no_dups = load_meta()
    l = len(index_images)
    import random
    r = random.random()
    index1 = int(r * l)
    if (pair):
        index2 = index1
    else:
        while True:
            r2 = random.random()
            index2 = int(l * r2)
            if (index2 != index1):
                break
    print(index1, index2)

    lpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index1) + '.jpg')
    rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index2) + '.jpg')

    image1 = cv2.imread(lpath)
    # cv2.imshow('image1',image1)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image2 = cv2.imread(rpath)
    # cv2.imshow('image2',image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image = [image1, image2]
    image = np.array(image)
    return image / 255


def dataAugmentation(path, dirpath, imageNo, count, sampleA):
    """
    :param path: test set or training set
    :param dirpath: relative path
    :param ImageNo: current ImageNumber
    :param count: total number of image pairs
    :param sampleA: left imageimageCount before dataaugmentation set
    :param sampleB: right imgae set
    :param width_res: restricted image width
    :param length_res: restricted image length
    :return:
    """
    for i in range(count):
        image = cv2.imread(str(sampleA[i]), 0)
        # 旋转180度
        rotated180 = rotate_bound(image, 180)
        cv2.imwrite("./img/" + str(path) + "temp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", image)
        cv2.imwrite("./img/" + str(path) + "temp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", rotated180)
        imageNo += 1
        # 向右平移10个像素 向下平移10个像素
        shifted = shift(image, 10, 10, image.shape[1], image.shape[0])
        cv2.imwrite("./img/" + str(path) + "temp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", rotated180)
        cv2.imwrite("./img/" + str(path) + "temp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", shifted)
        imageNo += 1
        # 均值模糊化处理
        blured = cv2.blur(image, (5, 5))
        cv2.imwrite("./img/" + str(path) + "temp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", shifted)
        cv2.imwrite("./img/" + str(path) + "temp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", blured)
        imageNo += 1
        # 随机区域裁剪
        x_axis = random.randint(0, image.shape[1] - 20)
        y_axis = random.randint(0, image.shape[0] - 20)
        croped = crop(image, x_axis, y_axis)
        cv2.imwrite("./img/" + str(path) + "temp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", blured)
        cv2.imwrite("./img/" + str(path) + "temp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", croped)
        imageNo += 1
        # 水平翻转
        fliped = cv2.flip(image, 1)
        cv2.imwrite("./img/" + str(path) + "temp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", croped)
        cv2.imwrite("./img/" + str(path) + "temp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", fliped)
        imageNo += 1
        # 原图和水平翻转
        cv2.imwrite("./img/" + str(path) + "temp/tempA" + dirpath + "img" + str(imageNo) + ".jpg", fliped)
        cv2.imwrite("./img/" + str(path) + "temp/tempB" + dirpath + "img" + str(imageNo) + ".jpg", image)
        imageNo += 1
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(rotated180)
        # plt.show()
        # plt.imshow(shifted)
        # plt.show()
        # plt.imshow(blured)
        # plt.show()
        # plt.imshow(croped)
        # plt.show()
        # plt.imshow(fliped)
        # plt.show()
        # print(image.shape, shift.shape, rotated90.shape, rotated45.shape, blured.shape, croped.shape, fliped.shape)

    return imageNo


def rotate(image, degree, xLength, yLength):
    center = (xLength // 2, yLength // 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    print("M shape ", M.shape)
    rotated = cv2.warpAffine(image, M, (xLength, xLength))
    print("rotated shape ", rotated.shape)
    return rotated


def shift(image, width, height, xLength, yLength):
    M = np.float32([[1, 0, width], [0, 1, height]])
    dst = cv2.warpAffine(image, M, (xLength, yLength))
    return dst


def crop(image, x_axis, y_axis):
    image[x_axis:x_axis + 10, y_axis:y_axis + 10] = 0
    return image


def rotate_bound(image, angle):
    """
    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img
def clearDir():
    # 覆盖图片目录
    dirpath = '/'
    del_list = ["./img/testtemp/tempA", "./img/testtemp/tempB", "./img/test/true", "./img/test/false",
                "./img/traintemp/tempA", "./img/traintemp/tempB", "./img/train/true", "./img/train/false"]
    del_list = [x + dirpath for x in del_list]
    for f in del_list:
        if not os.path.exists(f):
            os.makedirs(f)
        else:
            shutil.rmtree(f)
            os.makedirs(f)

if __name__ == "__main__":
    load_image_dif_size()
"""# 双通道 灰度图 中心采样和中心采样配对 原图片均为245*200
def load_image_same_size():
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()

    used_ids = []
    results = []

    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if (v['vote'] > 0.95):
            if uid not in used_ids:
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i
                })

    left = []
    right = []
    labels = []
    tmpTrue = [[], []]

    count = 0

    for i in range(len(results)):
        data = results[i]
        index = data['index']

        lpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
        count += 1
        tmpTrue[0].append(lpath)
        tmpTrue[0].append(rpath)

    tmpTrue = np.array(tmpTrue)

    # 覆盖图片目录
    dirpath = '/'
    del_list = ["./img/testtemp/tempA", "./img/testtemp/tempB", "./img/test/true", "./img/test/false",
                "./img/traintemp/tempA", "./img/traintemp/tempB", "./img/train/true", "./img/train/false"]
    del_list = [x + dirpath for x in del_list]
    for f in del_list:
        if not os.path.exists(f):
            os.makedirs(f)
        else:
            shutil.rmtree(f)
            os.makedirs(f)
    start = 0
    end = dataAugmentation("train", dirpath, start, len(tmpTrue[0]), tmpTrue[0])

    # 重新读取文件路径
    train_sampleA = []
    for i in range(start, end):
        train_sampleA.append("./img/traintemp/tempA/img" + str(i) + ".jpg")
    # 重新读取文件路径
    train_sampleB = []
    for i in range(start, end):
        train_sampleB.append("./img/traintemp/tempB/img" + str(i) + ".jpg")

    print("after re arrange: train_sampleB length is ", len(os.listdir("./img/traintemp/tempB" + dirpath)))
    print("after re arrange: train_sampleA length is ", len(os.listdir("./img/traintemp/tempA" + dirpath)))

    train_sampleTrue = np.array([train_sampleA, train_sampleB])
    train_sampleFalse = np.array([train_sampleA, train_sampleB])

    # 错误标签数据集
    np.random.shuffle(train_sampleFalse[0])
    np.random.shuffle(train_sampleFalse[1])
    # 检查是否匹配到了相同的图像
    length_train_sample = len(train_sampleFalse[0])
    for i in range(length_train_sample):
        left = int(train_sampleFalse[0][i].replace("./img/traintemp/tempA/img", "").replace(".jpg", ""))
        right = int(train_sampleFalse[1][i].replace("./img/traintemp/tempB/img", "").replace(".jpg", ""))
        # if abs(left - right) <= 10:
        if left / 6 == right / 6:
            print("left is ", left, "right is ", right)
            index = left - 6 if left > 6 else left + 6
            print("index is", index, "right is", right)
            train_sampleFalse[1][i] = train_sampleTrue[1][index]
    # 中心采样 下采样
    for i in range(end):
        lpath = train_sampleTrue[0][i]
        rpath = train_sampleTrue[1][i]
        handle_image_pair1(lpath, rpath, left, right)

    for i in range(end):
        labels.append(1)

    for i in range(end):
        lpath = train_sampleFalse[0][i]
        rpath = train_sampleFalse[1][i]
        handle_image_pair1(lpath, rpath, left, right)
    for i in range(end):
        labels.append(0)

    print(len(left), len(right), len(labels))

    left = np.array(left, dtype=object)
    right = np.array(right)

    # 打乱验证数据集
    np.random.seed(117)
    np.random.shuffle(left)
    np.random.seed(117)
    np.random.shuffle(right)
    np.random.seed(117)
    np.random.shuffle(labels)

    return left, right, labels
"""

def load_org_image(per_limit):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()
    print(len(votes.keys()))
    used_ids = []
    results = []
    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        print(uid)
        v = votes.get(uid, None)
        print(v)
        break

    count = 0
    for i, image in no_faces:
        if (count >= per_limit):
            break
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if (v['vote'] > 0.8):
            if (uid not in used_ids):
                count += 1
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i,
                    'type': 'no_faces'
                })

    count = 0
    for i, image in no_dups:
        if (count >= per_limit):
            break
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if (v['vote'] > 0.80):
            if (uid not in used_ids):
                count += 1
                used_ids.append(uid)
                results.append({
                    'uid': uid,
                    'index': i,
                    'type': 'no_faces'
                })

    centers = []
    labels = []
    tmpTrue = [[], []]
    tmpFalse = [[], []]

    count = 0

    for data in results:
        uid = data['uid']
        index = data['index']

        lpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
        count += 1
        if (count < len(results) / 2):
            tmpTrue[0].append(lpath)
            tmpTrue[1].append(lpath)
        else:
            tmpFalse[0].append(lpath)
            tmpFalse[1].append(lpath)

    tmpTrue = np.array(tmpTrue)
    tmpFalse = np.array(tmpFalse)

    np.random.shuffle(tmpFalse[0])
    np.random.shuffle(tmpFalse[1])

    truewf = open('./true-images.txt', 'w')
    for i in range(len(tmpTrue[0])):
        lpath = tmpTrue[0][i]
        rpath = tmpTrue[1][i]
        truewf.write('%s,%s\n' % (lpath, rpath))
        handle_image_pair(lpath, rpath, centers, None)
    truewf.close()

    for i in range(len(centers)):
        labels.append(1)
    print('True length:', len(labels))
    falsewf = open('./false-images.txt', 'w')
    for i in range(len(tmpFalse[0])):
        lpath = tmpFalse[0][i]
        rpath = tmpFalse[1][i]
        falsewf.write('%s,%s\n' % (lpath, rpath))
        handle_image_pair(lpath, rpath, centers, None)
    falsewf.close()
    print('False length:', len(centers) - len(labels))
    for i in range(len(centers) - len(labels)):
        labels.append(0)

    print(len(centers), len(labels))

    return centers, labels

def load_image(wid_res, length_res):
    votes = load_votes()
    index_images, no_faces, no_dups = load_meta()
    print(len(votes.keys()))
    used_ids = []
    results = []
    for i, image in index_images:
        uid = int(image.replace('.jpg', ''))
        v = votes.get(uid, None)
        if (v['vote'] > 0.8):
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
    tmpFalse = [[], []]

    count = 0

    for data in results:
        uid = data['uid']
        index = data['index']

        lpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/left', fix_index(index) + '.jpg')
        rpath = os.path.join('/mnt/mydata/DLData/Totally-Looks-Like-Data/right', fix_index(index) + '.jpg')
        count += 1
        if (count < len(results) / 2):
            tmpTrue[0].append(lpath)
            tmpTrue[1].append(lpath)
        else:
            tmpFalse[0].append(lpath)
            tmpFalse[1].append(lpath)

    tmpTrue = np.array(tmpTrue)
    tmpFalse = np.array(tmpFalse)

    np.random.shuffle(tmpFalse[0])
    np.random.shuffle(tmpFalse[1])

    print("length of tmpTrue are", np.array(tmpTrue).shape)
    for i in range(len(tmpTrue[0])):
        lpath = tmpTrue[0][i]
        rpath = tmpTrue[1][i]
        handle_image_pair(lpath, rpath, wid_res, length_res, centers, pyrds)

    for i in range(len(centers)):
        labels.append(1)

    for i in range(len(tmpFalse[0])):
        lpath = tmpFalse[0][i]
        rpath = tmpFalse[1][i]
        handle_image_pair(lpath, rpath, wid_res, length_res, centers, pyrds)

    for i in range(len(centers) - len(labels)):
        labels.append(0)

    print(len(centers), len(pyrds), len(labels))

    # 打乱验证数据集
    np.random.seed(116)
    np.random.shuffle(centers)
    np.random.seed(116)
    np.random.shuffle(pyrds)
    np.random.seed(116)
    np.random.shuffle(labels)

    return centers, pyrds, labels