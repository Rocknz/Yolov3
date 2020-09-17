import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.autograd import Variable


class Yolo(nn.Module):
    def __init__(self, s_index, e_index, S, C, img_size, lambda_coord, lambda_noobj):
        super(Yolo, self).__init__()
        self.S = S
        self.anchor = torch.Tensor(
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
        ).cuda()
        self.s_index = s_index
        self.e_index = e_index
        self.C = C
        self.img_size = img_size
        self.yolo_step = 3
        self.N = self.yolo_step * (5 + C)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def IOU(self, b1, b2):
        # [x1 y1 x2 y2]
        A = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
        B = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
        CM = (torch.min(b1[:, 2], b2[:, 2]) - torch.max(b1[:, 0], b2[:, 0]) + 1) * (
            torch.min(b1[:, 3], b2[:, 3]) - torch.max(b1[:, 1], b2[:, 1] + 1)
        )
        res = (CM) / (A + B - CM)
        del A, B, CM
        return res

    def IOU_index(self, w, h):
        anbox = torch.zeros([self.anchor.shape[0], 4]).cuda()
        anbox[:, 2] = self.anchor[:, 0]
        anbox[:, 3] = self.anchor[:, 1]
        whbox = torch.zeros([self.anchor.shape[0], 4]).cuda()
        whbox[:, 2] = w
        whbox[:, 3] = h

        iou = self.IOU(anbox, whbox)
        index = torch.argmax(iou)

        del iou, anbox, whbox
        return index

    def forward(self, batch_x, batch_box):
        loss = torch.Tensor([0]).cuda()
        for x, box in zip(batch_x, batch_box):
            box = torch.Tensor(box).cuda()

            check = torch.zeros(x.shape, dtype=torch.bool).cuda()
            check[0][:][:] = True
            check[(5 + self.C)][:][:] = True
            check[2 * (5 + self.C)][:][:] = True

            for abox in box:
                index = self.IOU_index(abox[3].item(), abox[4].item())
                if index >= self.s_index and index <= self.e_index:
                    now_index = index - self.s_index

                    div = self.img_size / self.S

                    index_x = int((abox[1] / div).item())
                    index_y = int((abox[2] / div).item())
                    alpha_x = ((abox[1] - index_x * div) / div).item()
                    alpha_y = ((abox[2] - index_y * div) / div).item()
                    check[now_index * (5 + self.C)][index_x][index_y] = False

                    now_x = x.select(2, index_y)
                    now_x = now_x.select(1, index_x)

                    res_alpha_x = torch.sigmoid(now_x[now_index * (5 + self.C) + 1])
                    res_alpha_y = torch.sigmoid(now_x[now_index * (5 + self.C) + 2])
                    # res_alpha_x = now_x[now_index * (5 + self.C) + 1]
                    # res_alpha_y = now_x[now_index * (5 + self.C) + 2]
                    res_w = self.anchor[index][0] * torch.exp(now_x[now_index * (5 + self.C) + 3])
                    res_h = self.anchor[index][1] * torch.exp(now_x[now_index * (5 + self.C) + 4])

                    rect = [res_alpha_x.item(), res_alpha_y.item(), res_w.item(), res_h.item()]
                    # rect = [res_alpha_x, res_alpha_y, res_w, res_h]
                    b1 = torch.Tensor(
                        [
                            [
                                rect[0] * div - rect[2] / 2,
                                rect[1] * div - rect[3] / 2,
                                rect[0] * div + rect[2] / 2,
                                rect[1] * div + rect[3] / 2,
                            ]
                        ]
                    ).cuda()
                    b2 = torch.Tensor(
                        [
                            [
                                alpha_x * div - abox[3].item() / 2,
                                alpha_y * div - abox[4].item() / 2,
                                alpha_x * div + abox[3].item() / 2,
                                alpha_y * div + abox[4].item() / 2,
                            ]
                        ]
                    ).cuda()
                    iou = self.IOU(b1, b2)
                    hot_enco = torch.zeros([self.C]).cuda()
                    hot_enco[int(abox[0].item())] = 1
                    label = torch.sigmoid(
                        now_x[now_index * (5 + self.C) + 5 : (now_index + 1) * (5 + self.C)]
                    )

                    val = torch.Tensor([alpha_x, alpha_y, abox[3], abox[4]]).cuda()
                    loss = loss + self.lambda_coord * self.mse(
                        now_x[now_index * (5 + self.C)], iou[0]
                    )
                    loss = loss + self.bce(label, hot_enco)
                    loss = loss + self.mse(res_alpha_x, val[0])
                    loss = loss + self.mse(res_alpha_y, val[1])
                    loss = loss + self.mse(res_w, val[2])
                    loss = loss + self.mse(res_h, val[3])

                    del val, hot_enco, label, iou, b1, b2

            val = torch.zeros(x[check].shape).cuda()
            loss = loss + self.lambda_noobj * self.mse(x[check], val)

            del val, check, box

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
        y = self.batch_mid(self.res_conv1(x))
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
        #     nn.BatchNorm2d(self.B * (5 + self.C)),
        #     nn.LeakyReLU(),
        # )
        self.yolo1 = Yolo(6, 8, 21, self.C, self.img_size, lambda_coord, lambda_noobj)
        self.yolo2 = Yolo(3, 5, 42, self.C, self.img_size, lambda_coord, lambda_noobj)
        self.yolo3 = Yolo(0, 2, 84, self.C, self.img_size, lambda_coord, lambda_noobj)

    def forward(self, x, boxes):
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
        loss1 = self.yolo1(y1, boxes)

        # 42
        y2_1 = self.y2c1(y1_1)
        x2 = self.merge2(x2)
        y2_1 = torch.cat((y2_1, x2), 1)
        y2_2 = self.y2c2(y2_1)
        y2 = self.y2c3(y2_2)
        loss2 = self.yolo2(y2, boxes)

        # 84
        y3_1 = self.y3c1(y2_2)
        x3 = self.merge3(x3)
        y3_1 = torch.cat((y3_1, x3), 1)
        y3 = self.y3c2(y3_1)
        y3 = self.y3c3(y3)
        loss3 = self.yolo3(y3, boxes)

        loss = loss1 + loss2 + loss3

        # return loss, yolo1, yolo2, yolo3, y3_3
        del x, x1, x2, x3, y1_1, y1, y2_1, y2_2, y2, y3_1, y3
        return loss

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        try:
            self.load_state_dict(torch.load(self.path))
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


if __name__ == "__main__":

    S = 7
    B = 2
    C = 92
    img_size = 672

    dataDir = "datas/CoCo"
    dataType = "val2017"
    annFile = "{}/annotations_trainval2017/annotations/instances_{}.json".format(dataDir, dataType)
    print(annFile)
    data = DataLoad(dataDir, dataType, annFile, C, img_size=img_size)
    data.load_val_start(data_num=2, batch_size=2, worker=1)

    time.sleep(5)
    yolo = YoloV3(S, B, C, lambda_coord=5, lambda_noobj=0.5).cuda()
    image, boxes = data.load_val()

    # lr = 0.00001
    # optimizer = torch.optim.Adam(yolo.parameters(), lr=lr)
    optimizer = torch.optim.Adam(yolo.parameters())

    image = torch.from_numpy(image).float().cuda()
    for i in range(300):
        optimizer.zero_grad()
        loss = yolo(image, boxes)
        print(loss.item())

        loss.backward()
        optimizer.step()

    loss = yolo(image, boxes)
    print(loss.item())
