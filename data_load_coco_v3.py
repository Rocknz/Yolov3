# list up all folders & data names!

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datas.CoCo.cocoapi.PythonAPI.pycocotools.coco import COCO
from os import listdir
from os.path import isfile, isdir, join
from torch.utils import data
from PIL import Image
import numpy as np
import random
import time
import cv2
import queue
import json
import math
from threading import Thread, Lock


class DataLoad:
    def __init__(self, dataDir, dataType, annFile, C, img_size=640):
        # anchor, S

        self.dataDir = dataDir
        self.dataType = dataType

        self.coco = COCO(annFile)
        self.files = self.coco.getImgIds()
        self.files.sort()

        # self.anchor = anchor
        # self.S = S
        self.C = C

        self.img_size = img_size

    def size_files(self):
        return len(self.files)

    def load_train_start(self, data_num=10, batch_size=4, worker=4):
        self.data_num = data_num
        self.batch_size = batch_size

        self.train_data = queue.Queue()
        self.train_data_mutex = Lock()

        self.threads = [Thread(target=self.load_train_thread) for i in range(worker)]

        print("Threads start!")
        self.thread_running = True

        # self.threads = []
        # self.load_train_thread()

        for thr in self.threads:
            thr.start()
        print("Threads working!")

    def load_train(self):
        while True:
            self.train_data_mutex.acquire()
            if self.train_data.qsize() > 0:
                res = self.train_data.get()
                self.train_data_mutex.release()
                return res
            else:
                self.train_data_mutex.release()
                time.sleep(0.1)
                print("waiting data..")
                continue

    def load_train_thread(self):
        while self.thread_running:
            num = self.train_data.qsize()
            if num >= self.data_num:
                time.sleep(1)
                continue

            images_np = np.empty((0, 3, self.img_size, self.img_size), float)
            boundingboxes = []

            iter = 0
            while iter < self.batch_size:
                try:
                    file_index = random.randint(0, len(self.files))
                    image_name = self.coco.loadImgs(self.files[file_index])

                    anno_id = self.coco.getAnnIds(imgIds=image_name[0]["id"])
                    anns = self.coco.loadAnns(anno_id)

                    img_path = self.dataDir + "/" + self.dataType + "/" + image_name[0]["file_name"]
                    image = cv2.imread(img_path)
                    # cv2.imshow('img',image)
                    # cv2.waitKey(0)

                    image_np = np.asarray(image)

                    boundingbox = []
                    for annos in anns:
                        boundingbox.append([annos["category_id"]] + annos["bbox"][:])
                    # if image_np.shape[0] > image_np.shape[1] :
                    #     types = '640x640'
                    # else:
                    #     types = '480x640'
                    #     image_np = image_np.transpose(1, 0, 2)

                    # Change all dimensions => [640, 640]

                    image_np = image_np.transpose(2, 1, 0)
                    image_np = np.pad(
                        image_np,
                        (
                            (0, 0),
                            (0, self.img_size - image_np.shape[1]),
                            (0, self.img_size - image_np.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )

                    # cv2.imshow('img',image)
                    # cv2.waitKey(0)
                    # cv2.imshow('img1',image_np[0].transpose(1, 2, 0))
                    # cv2.waitKey(0)

                    image_np = image_np[np.newaxis, ...]
                    images_np = np.append(images_np, image_np, axis=0)

                    boundingboxes.append(boundingbox)

                    iter = iter + 1

                except Exception as e:
                    print(e)
                    continue

            boundingboxes = self.encode(boundingboxes)
            datas = (images_np, boundingboxes)
            self.train_data_mutex.acquire()
            self.train_data.put(datas)
            # print(self.train_data.qsize())
            self.train_data_mutex.release()

    def load_val_start(self, data_num=10, batch_size=4, worker=1):
        self.data_num = data_num
        self.batch_size = batch_size

        self.val_data = queue.Queue()
        self.val_data_mutex = Lock()

        self.threads = [Thread(target=self.load_val_thread) for i in range(worker)]
        self.val_cnt = 0

        print("Threads start!")
        self.thread_running = True

        for thr in self.threads:
            thr.start()
        print("Threads working!")

    def load_val(self):
        while True:
            self.val_data_mutex.acquire()
            if self.val_data.qsize() > 0:
                res = self.val_data.get()
                self.val_data_mutex.release()
                return res
            else:
                self.val_data_mutex.release()
                time.sleep(0.1)
                print("waiting data..")
                continue

    def load_val_thread(self):
        while self.thread_running:
            num = self.val_data.qsize()
            if num >= self.data_num:
                time.sleep(1)
                continue

            images_np = np.empty((0, 3, self.img_size, self.img_size), float)
            boundingboxes = []

            iter = 0
            while iter < self.batch_size:
                try:
                    image_name = self.coco.loadImgs(self.files[self.val_cnt])
                    self.val_cnt += 1
                    if self.val_cnt >= self.data_num * self.batch_size:
                        self.val_cnt = 0

                    anno_id = self.coco.getAnnIds(imgIds=image_name[0]["id"])
                    anns = self.coco.loadAnns(anno_id)

                    img_path = self.dataDir + "/" + self.dataType + "/" + image_name[0]["file_name"]
                    image = cv2.imread(img_path)
                    # cv2.imshow('img',image)
                    # cv2.waitKey(0)

                    image_np = np.asarray(image)

                    boundingbox = []
                    for annos in anns:
                        boundingbox.append([annos["category_id"]] + annos["bbox"][:])
                    # if image_np.shape[0] > image_np.shape[1] :
                    #     types = '640x640'
                    # else:
                    #     types = '480x640'
                    #     image_np = image_np.transpose(1, 0, 2)

                    # Change all dimensions => [640, 640]

                    image_np = image_np.transpose(2, 1, 0)
                    image_np = np.pad(
                        image_np,
                        (
                            (0, 0),
                            (0, self.img_size - image_np.shape[1]),
                            (0, self.img_size - image_np.shape[2]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )

                    # cv2.imshow('img',image)
                    # cv2.waitKey(0)
                    # cv2.imshow('img1',image_np[0].transpose(1, 2, 0))
                    # cv2.waitKey(0)

                    image_np = image_np[np.newaxis, ...]
                    images_np = np.append(images_np, image_np, axis=0)

                    boundingboxes.append(boundingbox)

                    iter = iter + 1

                except Exception as e:
                    print(e)
                    continue

            boundingboxes = self.encode(boundingboxes)
            datas = (images_np, boundingboxes)
            self.val_data_mutex.acquire()
            self.val_data.put(datas)
            # print(self.train_data.qsize())
            self.val_data_mutex.release()

    def close(self):
        self.thread_running = False
        print("Threads destroyed.")

    def encode(self, boundingboxes):
        boxes = np.asarray([])
        n_index = 0

        for raw_boxes in boundingboxes:
            box = []
            for bb in raw_boxes:
                bb[1] = bb[1] + bb[3] / 2
                bb[2] = bb[2] + bb[4] / 2
                box.append(bb)
            box = np.asarray(box)

            if box.shape[0] == 0:
                continue

            index = np.zeros([box.shape[0], 1]) + n_index
            n_index += 1

            add_boxes = box[:, :]
            add_boxes = np.append(index, box, axis=1)
            if boxes.shape[0] == 0:
                boxes = np.array(add_boxes)
            else:
                boxes = np.append(boxes, add_boxes, axis=0)

        return boxes
        # return boundingboxes


import signal
import sys


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    global data
    data.close()


if __name__ == "__main__":
    TRAIN = False
    global data

    # anchor = [[10, 13], [16, 30], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    # S = [81, 81, 81, ]

    if TRAIN:
        dataDir = "datas/CoCo"
        dataType = "train2017"
        annFile = "{}/annotations_trainval2017/annotations/instances_{}.json".format(
            dataDir, dataType
        )
        img_size = 640
        B = 2
        C = 92
        data = DataLoad(dataDir, dataType, annFile, C=C, img_size=img_size)
        data.load_train_start(data_num=1, batch_size=1, worker=1)
    else:
        dataDir = "datas/CoCo"
        dataType = "val2017"
        annFile = "{}/annotations_trainval2017/annotations/instances_{}.json".format(
            dataDir, dataType
        )
        img_size = 640
        B = 2
        C = 92
        data = DataLoad(dataDir, dataType, annFile, C=C, img_size=img_size)
        data.load_val_start(data_num=1, batch_size=1, worker=1)

    while True:
        # signal.signal(signal.SIGINT, signal_handler)
        # time.sleep(10)
        if TRAIN:
            img, box = data.load_train()
        else:
            img, box = data.load_val()

        import cv2, time, torch

        image = img[0].transpose(2, 1, 0).astype("uint8")
        print(image.shape)

        t_start = time.clock()

        box = torch.Tensor(box).cuda()
        # draw boxes
        # tensor pratice
        print(box.shape)
        for bb in range(box.shape[0]):
            x = box[bb][2]
            y = box[bb][3]
            w = box[bb][4]
            h = box[bb][5]

            image = cv2.UMat(image).get()
            start = (x - w / 2, y - h / 2)
            end = (x + w / 2, y + h / 2)
            color = (255, 0, 0)
            cv2.rectangle(image, start, end, color, 2)

        print(box)
        t_end = time.clock()
        print(t_end - t_start)
        # for boxes in box[0]:
        #     x = boxes[1]
        #     y = boxes[2]
        #     g_x = boxes[3]
        #     g_y = boxes[4]
        #     w = (boxes[5] * boxes[5]) * img_size
        #     h = (boxes[6] * boxes[6]) * img_size
        #     x_c = ((x + g_x) / S) * img_size
        #     y_c = ((y + g_y) / S) * img_size

        #     image = cv2.UMat(image).get()
        #     start = (int(x_c - w/2), int(y_c - h/2))
        #     end = (int(x_c + w/2), int(y_c + h/2))
        #     color = (255,0,0)
        #     cv2.rectangle(image, start, end, color, 2)
        #     print((start, end))

        cv2.imshow("img", image)
        cv2.waitKey(0)
