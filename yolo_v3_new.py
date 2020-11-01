import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.autograd import Variable
import os


class Yolo(nn.Module):
    def __init__(self, s_index, e_index, S, C, img_size, lambda_coord, lambda_noobj):
        super(Yolo, self).__init__()
        self.S = S
        self.anchor = Variable(
            torch.Tensor(
                [
                    [10, 13],
                    [16, 30],
                    [33, 23],
                    [30, 61],
                    [62, 45],
                    [59, 119],
                    [116, 90],
                    [156, 198],
                    [373, 326],
                ]
            ),
            requires_grad=False,
        ).cuda()
        self.s_index = s_index
        self.e_index = e_index
        self.C = C
        self.img_size = img_size
        self.yolo_step = 3
        self.N = self.yolo_step * (5 + C)
        # self.bce = lambda a, b: torch.pow(a - b, 2).sum()
        # self.mse = lambda a, b: torch.pow(a - b, 2).sum()
        # self.mse = nn.MSELoss()
        # self.bce = nn.MSELoss()
        # self.mse = nn.MSELoss()
        # self.bce = nn.BCELoss()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def mse(self, A, B):
        return ((A - B) ** 2).sum()

    def bce(self, A, B):
        return ((A - B) ** 2).sum()

    def IOU(self, b1, b2):
        # [x1 y1 x2 y2]
        A = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
        B = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
        CM = (torch.min(b1[:, 2], b2[:, 2]) - torch.max(b1[:, 0], b2[:, 0]) + 1) * (
            torch.min(b1[:, 3], b2[:, 3]) - torch.max(b1[:, 1], b2[:, 1]) + 1
        )
        res = (CM) / (A + B - CM)

        res_sat = res < 0
        res[res_sat] = 0

        del A, B, CM, res_sat
        return res

    def IOU_batch(self, b1, b2):
        # [x1 y1 x2 y2]
        A = (b1[:, :, 2] - b1[:, :, 0] + 1) * (b1[:, :, 3] - b1[:, :, 1] + 1)
        B = (b2[:, :, 2] - b2[:, :, 0] + 1) * (b2[:, :, 3] - b2[:, :, 1] + 1)
        CM = (torch.min(b1[:, :, 2], b2[:, :, 2]) - torch.max(b1[:, :, 0], b2[:, :, 0]) + 1) * (
            torch.min(b1[:, :, 3], b2[:, :, 3]) - torch.max(b1[:, :, 1], b2[:, :, 1]) + 1
        )
        res = (CM) / (A + B - CM)

        res_sat = res < 0
        res[res_sat] = 0

        del A, B, CM, res_sat
        return res

    def IOU_index(self, w, h):
        anbox = Variable(torch.zeros([self.anchor.shape[0], 4])).cuda()
        anbox[:, 2] = self.anchor[:, 0]
        anbox[:, 3] = self.anchor[:, 1]
        whbox = Variable(torch.zeros([w.shape[0], 4])).cuda()
        whbox[:, 2] = w[:]
        whbox[:, 3] = h[:]

        anbox = anbox.unsqueeze(0).repeat([w.shape[0], 1, 1])
        whbox = whbox.unsqueeze(1).repeat([1, self.anchor.shape[0], 1])

        iou = self.IOU_batch(anbox, whbox)
        index = torch.argmax(iou, 1)

        del iou, anbox, whbox
        return index

    def forward(self, x, n_box, n_index):
        # x = [iou, x_a, y_a, w, h]
        torch.cuda.set_device(0)
        x = torch.sigmoid(x)

        loss = Variable(torch.zeros([1]), requires_grad=True).cuda()
        if torch.isinf(x).max() or torch.isnan(x).max():
            print("forward inf or nan exists!")
            raise Exception("FORWARD")

        check = Variable(torch.zeros(x.shape, dtype=torch.bool), requires_grad=False).cuda()
        check[:, 0, :, :] = True
        check[:, (5 + self.C), :, :] = True
        check[:, 2 * (5 + self.C), :, :] = True

        if n_box.shape[0] != 0:
            val = self.s_index <= n_index[:]
            val = val * (n_index[:] <= self.e_index)
            box = n_box[val]

            if box.shape[0] != 0:
                index = n_index[val]

                box_size = box.shape[0]
                now_index = index[:] - self.s_index

                div = self.img_size / self.S
                batch_index = box[:, 0].long()
                index_x = (box[:, 2] / div).long()
                index_y = (box[:, 3] / div).long()
                alpha_x = (box[:, 2] - index_x[:] * div) / div
                alpha_y = (box[:, 3] - index_y[:] * div) / div
                check[batch_index[:], now_index[:] * (5 + self.C), index_x[:], index_y[:]] = False

                # start this

                now_x = x[batch_index, 0 : (3 * (5 + self.C)), index_x, index_y]
                now_x = now_x.transpose(1, 0)
                box_iter = torch.Tensor(range(box_size)).long().cuda()

                res_alpha_x = now_x[now_index[:] * (5 + self.C) + 1, box_iter]
                res_alpha_y = now_x[now_index[:] * (5 + self.C) + 2, box_iter]

                # w = now_x[now_index * (5 + self.C) + 3, box_iter]
                # h = now_x[now_index * (5 + self.C) + 4, box_iter]

                res_w = self.anchor[index, 0] * torch.exp(
                    4 * now_x[now_index * (5 + self.C) + 3, box_iter]
                    - 2
                    # w
                )
                res_h = self.anchor[index, 1] * torch.exp(
                    4 * now_x[now_index * (5 + self.C) + 4, box_iter]
                    - 2
                    # h
                )

                # rect = Variable(torch.zeros([box_size, 4])).cuda()
                # rect[:, 0] = res_alpha_x
                # rect[:, 1] = res_alpha_y
                # rect[:, 2] = res_w
                # rect[:, 3] = res_h

                # b1 = Variable(torch.zeros([box_size, 4])).cuda()
                # b1[:, 0] = rect[:, 0] * div - rect[:, 2] / 2
                # b1[:, 1] = rect[:, 1] * div - rect[:, 3] / 2
                # b1[:, 2] = rect[:, 0] * div + rect[:, 2] / 2
                # b1[:, 3] = rect[:, 1] * div + rect[:, 3] / 2

                # b2 = Variable(torch.zeros([box_size, 4])).cuda()
                # b2[:, 0] = alpha_x * div - box[:, 4] / 2
                # b2[:, 1] = alpha_y * div - box[:, 5] / 2
                # b2[:, 2] = alpha_x * div + box[:, 4] / 2
                # b2[:, 3] = alpha_y * div + box[:, 5] / 2

                # iou = self.IOU(b1, b2)
                hot_enco = Variable(torch.zeros([box_size, self.C])).cuda()

                hot_enco[box_iter, box[:, 1].long()] = 1
                label_range = torch.Tensor(range(self.C)).cuda().long().repeat([box_size]) + 5
                now_index_val = now_index.unsqueeze(1).repeat([1, self.C]).reshape(
                    [self.C * box_size]
                ) * (5 + self.C)
                label_range = label_range + now_index_val
                label_range = label_range.long()

                box_size_range = (
                    torch.Tensor(range(box_size))
                    .cuda()
                    .long()
                    .unsqueeze(1)
                    .repeat([1, self.C])
                    .reshape([self.C * box_size])
                )
                label = now_x[label_range, box_size_range].reshape([box_size, self.C])
                # label = torch.sigmoid(
                #     now_x[label_range, box_size_range].reshape([box_size, self.C])
                # )

                iou = torch.ones([box_size]).cuda()
                loss = loss + self.lambda_coord * self.bce(
                    now_x[now_index * (5 + self.C), box_iter], iou[:]
                )
                loss = loss + self.bce(label, hot_enco)
                loss = loss + self.bce(res_alpha_x, alpha_x)
                loss = loss + self.bce(res_alpha_y, alpha_y)
                # this to size down.
                loss = loss + self.mse(res_w / self.img_size, box[:, 4] / self.img_size)
                loss = loss + self.mse(res_h / self.img_size, box[:, 5] / self.img_size)

                # loss = loss + self.mse(w, torch.log(box[:, 4] / self.anchor[index, 0] + 1e-16))
                # loss = loss + self.mse(h, torch.log(box[:, 5] / self.anchor[index, 1] + 1e-16))

                del (
                    val,
                    hot_enco,
                    label,
                    iou,
                    label_range,
                    box_size_range,
                    now_index_val,
                    now_index,
                    index,
                )

        # n_x = torch.sigmoid(x[check])
        n_x = x[check]
        val = Variable(torch.zeros(n_x.shape)).cuda()
        loss = loss + self.lambda_noobj * self.bce(n_x, val)

        del n_x, val, check

        if torch.isinf(loss):
            print("inf!!")
            loss = loss
            raise Exception("INF")

        if torch.isnan(loss):
            print("nan!!")
            loss = loss
            raise Exception("NAN")

        return loss


class Residual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Residual, self).__init__()
        self.res_conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=False
        )
        self.res_conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.leakyrelu = nn.LeakyReLU()
        self.batch_mid = nn.BatchNorm2d(mid_channels)
        self.batch_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.res_conv1(x)
        y = self.batch_mid(y)
        y = self.leakyrelu(y)
        y = self.res_conv2(y)
        z = self.leakyrelu(x + y)
        z = self.batch_out(z)
        return z


class YoloV3(nn.Module):
    def __init__(self, S, B, C, img_size=672, lambda_coord=5, lambda_noobj=0.5):
        super(YoloV3, self).__init__()
        self.path = "yolo_model"
        self.B = B
        self.C = C
        self.S = S
        self.yolo_step = 3
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.img_size = img_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        # 336
        self.res1 = Residual(64, 32, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        # 168
        self.res2 = nn.Sequential(
            Residual(128, 64, 128),
            Residual(128, 64, 128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        # 84
        self.res3 = nn.Sequential(
            Residual(256, 128, 256),
            Residual(256, 128, 256),
            Residual(256, 128, 256),
            Residual(256, 128, 256),
            Residual(256, 128, 256),
            Residual(256, 128, 256),
            Residual(256, 128, 256),
            Residual(256, 128, 256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        # 42
        self.res4 = nn.Sequential(
            Residual(512, 256, 512),
            Residual(512, 256, 512),
            Residual(512, 256, 512),
            Residual(512, 256, 512),
            Residual(512, 256, 512),
            Residual(512, 256, 512),
            Residual(512, 256, 512),
            Residual(512, 256, 512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 21
        self.res5 = nn.Sequential(
            Residual(1024, 512, 1024),
            Residual(1024, 512, 1024),
            Residual(1024, 512, 1024),
            Residual(1024, 512, 1024),
        )
        self.y1c1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.y1c2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, self.yolo_step * (5 + self.C), kernel_size=1, stride=1, bias=False),
        )
        self.y2c1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.y2c2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.y2c3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, self.yolo_step * (5 + self.C), kernel_size=1, stride=1, bias=False),
        )

        self.y3c1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.merge3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.y3c2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.y3c3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, self.yolo_step * (5 + self.C), kernel_size=1, stride=1, bias=False),
        )

        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(1024, self.B * (5 + self.C), kernel_size=1, stride=1, padding=0, bias=False),
        #     #nn.BatchNorm2d(self.B * (5 + self.C)),
        #     nn.LeakyReLU(),
        # )
        self.yolo1 = Yolo(6, 8, 21, self.C, self.img_size, lambda_coord, lambda_noobj)
        self.yolo2 = Yolo(3, 5, 42, self.C, self.img_size, lambda_coord, lambda_noobj)
        self.yolo3 = Yolo(0, 2, 84, self.C, self.img_size, lambda_coord, lambda_noobj)

        # self.yolo1_prev = Yolo_prev(6, 8, 21, self.C, self.img_size, lambda_coord, lambda_noobj)
        # self.yolo2_prev = Yolo_prev(3, 5, 42, self.C, self.img_size, lambda_coord, lambda_noobj)
        # self.yolo3_prev = Yolo_prev(0, 2, 84, self.C, self.img_size, lambda_coord, lambda_noobj)

    def forward(self, x, boxes):
        index = []
        if boxes.shape[0] != 0:
            index = self.yolo1.IOU_index(boxes[:, 4], boxes[:, 5])

        # 336
        x = self.conv1(x)
        x = self.res1(x)
        # 168
        x = self.conv2(x)
        x = self.res2(x)
        # 84
        x3 = self.conv3(x)
        x = self.res3(x3)
        # 42
        x2 = self.conv4(x)
        x = self.res4(x2)

        # 21
        x = self.conv5(x)
        x1 = self.res5(x)

        y1_1 = self.y1c1(x1)
        y1 = self.y1c2(y1_1)
        loss1 = self.yolo1(y1, boxes, index)

        # 42
        y2_1 = self.y2c1(y1_1)
        x2 = self.merge2(x2)
        y2_1 = torch.cat((y2_1, x2), 1)
        y2_2 = self.y2c2(y2_1)
        y2 = self.y2c3(y2_2)
        loss2 = self.yolo2(y2, boxes, index)

        # 84
        y3_1 = self.y3c1(y2_2)
        x3 = self.merge3(x3)
        y3_1 = torch.cat((y3_1, x3), 1)
        y3 = self.y3c2(y3_1)
        y3 = self.y3c3(y3)
        loss3 = self.yolo3(y3, boxes, index)

        loss = loss1 + loss2 + loss3

        # del x1, x2, x3, y1_1, y2_1, y2_2, y3_1, index, loss1, loss2, loss3
        return loss, y1, y2, y3

    def save(self, name="yolo_model"):
        torch.save(self.state_dict(), name)

    def load(self, name="yolo_model"):
        try:
            self.load_state_dict(torch.load(name))
            return True
        except:
            print("load_error!")
            return False


# from ..data_load_coco import DataLoad
from data_load_coco_v3 import DataLoad
import time


def test():
    a = torch.randint(0, 100, (4, 4))
    b = torch.randint(0, 100, (4, 4))
    val = a * b
    print(a)
    print(b)
    print(val)


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        print("run cuda")
        torch.cuda.set_device(0)

    if __name__ == "__main__":
        S = 7
        B = 2
        C = 92
        img_size = 672

        dataDir = "datas/CoCo"
        dataType = "val2017"
        annFile = "{}/annotations_trainval2017/annotations/instances_{}.json".format(
            dataDir, dataType
        )
        print(annFile)
        data = DataLoad(dataDir, dataType, annFile, C, img_size=img_size)
        data.load_val_start(data_num=10, batch_size=1, worker=1)

        time.sleep(5)
        yolo = YoloV3(S, B, C, lambda_coord=5, lambda_noobj=0.5).cuda()
        raw_image, batch_boxes = data.load_val()

        # lr = 0.001
        lr = 0.001
        optimizer = torch.optim.Adam(yolo.parameters(), lr=lr)

        start = time.time()
        image = torch.from_numpy(raw_image).float().cuda()
        boxes = torch.from_numpy(batch_boxes).float().cuda()

        for i in range(200):
            try:
                optimizer.zero_grad()
                (loss, y1, y2, y3) = yolo(image, boxes)
                print(loss.item())

                loss.backward()
                optimizer.step()
            except Exception as e:
                print("Error occur!.", e)
                pass

        end = time.time()
        print(end - start)
        (loss, y1, y2, y3) = yolo(image, boxes)
        print(loss.item())

        # import make_yolo_from_box
        # y1, y2, y3 = make_yolo_from_box(boxes)

        from val_visualize import print_box
        import cv2

        for i in range(1):
            image = raw_image[i] * 255.0
            image = image.transpose(2, 1, 0).astype("uint8")
            image = cv2.UMat(image).get()

            print_box(image, y1, img_size, 6, batch_index=i)
            print_box(image, y2, img_size, 3, batch_index=i)
            print_box(image, y3, img_size, 0, batch_index=i)

            cv2.imshow("img", image)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()