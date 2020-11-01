from data_load_coco_v3 import DataLoad
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from yolo_v3_new import YoloV3
import os
from os import path


def validation():
    global model, S, B, Category_num, min
    global val_data_num
    global data_v

    loss_tot = 0.0
    data_num = val_data_num

    with torch.no_grad():
        for i in range(data_num):

            img, anno = data_v.load_val()
            torch.cuda.set_device(0)
            img = torch.from_numpy(img).float().cuda()
            anno = torch.from_numpy(anno).float().cuda()

            torch.cuda.set_device(0)
            (loss, y1, y2, y3) = model(img, anno)
            loss_tot += loss.item()

            del loss, img, anno
            # torch.cuda.empty_cache()

    loss_tot /= data_num
    print("validation loss : {}".format(loss_tot))
    return loss_tot


def train(lr_init):
    global model, S, B, Category_num

    min_val = 1e10
    try:
        if path.exists("min_val.txt"):
            files = open("min_val.txt", "r")
            min_val = float(files.read())
            files.close()
    except:
        pass

    epochs = 1000
    data_num = 500

    for epoch in range(epochs):
        lr = lr_init
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        i = 0
        while True:
            img, anno = data.load_train()
            optimizer.zero_grad()

            # not too much box for memory ..
            if anno.shape[0] > 100:
                continue

            torch.cuda.set_device(0)
            img = torch.from_numpy(img).float().cuda()
            anno = torch.from_numpy(anno).float().cuda()

            torch.cuda.set_device(0)
            (loss, y1, y2, y3) = model(img, anno)
            loss.backward()

            optimizer.step()
            if i % 50 == 0:
                print("1st loss {}".format(loss.item()))
                del loss
                with torch.no_grad():
                    torch.cuda.set_device(0)
                    (loss, y1, y2, y3) = model(img, anno)
                print("2nd loss {}".format(loss.item()))
                model.save()

            i = i + 1
            if i > data_num:
                break

            del loss, img, anno
            # torch.cuda.empty_cache()

        print("epoch : {} end".format(epoch))

        loss_val = validation()
        if loss_val < min_val:
            min_val = loss_val
            files = open("min_val.txt", "w")
            files.write(str(min_val))
            model.save("yolo_min_model")
            print("min val model saved!")
            files.close()

        model.save()

        del loss_val


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def snipp():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    # data = torch.randn([5, 32, 672, 672], dtype=torch.float, device="cuda", requires_grad=True)
    # net = torch.nn.Conv2d(
    #     32, 64, kernel_size=[3, 3], padding=[1, 1], stride=[2, 2], dilation=[1, 1], groups=1
    # )
    # net = net.cuda().float()
    # out = net(data)
    # out.backward(torch.randn_like(out))
    torch.cuda.synchronize()


def main():
    # snipp()
    B = 2
    Category_num = 92
    img_size = 672

    global val_data_num, data, data_v

    dataDir = "datas/CoCo"
    dataType = "train2017"
    annFile = "{}/annotations_trainval2017/annotations/instances_{}.json".format(dataDir, dataType)
    data = DataLoad(dataDir, dataType, annFile, Category_num, img_size=img_size)
    data.load_train_start(data_num=10, batch_size=3, worker=10)

    dataDir_v = "datas/CoCo"
    dataType_v = "val2017"
    annFile_v = "{}/annotations_trainval2017/annotations/instances_{}.json".format(
        dataDir_v, dataType_v
    )

    val_data_num = 100
    data_v = DataLoad(dataDir_v, dataType_v, annFile_v, Category_num, img_size=img_size)
    data_v.load_val_start(data_num=val_data_num, batch_size=1, worker=1)
    # torch.backends.cudnn.enabled = False

    global model

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    torch.cuda.set_device(0)
    torch.cuda.memory.empty_cache()
    torch.cuda.empty_cache()

    S = 7
    model = YoloV3(S, B, Category_num, lambda_coord=5, lambda_noobj=0.5)

    print(torch.backends.cudnn.benchmark)
    torch.cuda.set_device(0)
    try:
        with torch.autograd.set_detect_anomaly(True):

            global min_loss

            if model.load():
                print("model load end!!")
                model.cuda()
                # min_loss = validation()
            else:
                min_loss = 1e100
                model.cuda()
                # min_loss = validation()

            import time

            # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = False

            torch.cuda.set_device(0)
            train(0.001)

    except Exception as e:
        print("Error occur!.", e)
        data.close()
        data_v.close()
        sys.exit(0)


if __name__ == "__main__":
    main()