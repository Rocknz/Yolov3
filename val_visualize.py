import torch
from yolo_v3_new import YoloV3
from data_load_coco_v3 import DataLoad
import cv2

B = 2
Category_num = 92
img_size = 672
S = 7

dataDir_v = "datas/CoCo"
dataType_v = "val2017"
annFile_v = "{}/annotations_trainval2017/annotations/instances_{}.json".format(
    dataDir_v, dataType_v
)
data_v = DataLoad(dataDir_v, dataType_v, annFile_v, Category_num, img_size=img_size)
data_v.load_val_start(data_num=100, batch_size=1, worker=1)

model = YoloV3(S, B, Category_num, lambda_coord=5, lambda_noobj=0.5)


def print_box(image, box, img_size, start_index, batch_index=0):
    box = torch.sigmoid(box)
    anchor = torch.Tensor(
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
    box_size = box.shape[2]
    box_len = img_size / box_size
    for i in range(box_size):
        for j in range(box_size):
            nbox = box[batch_index, :, i, j]
            for indx in range(3):
                start = indx * (5 + Category_num)
                if nbox[start] >= 0.1:
                    alpha_x = nbox[start + 1] * box_len
                    alpha_y = nbox[start + 2] * box_len
                    w = anchor[start_index + indx, 0] * torch.exp(4 * nbox[start + 3] - 2)
                    h = anchor[start_index + indx, 1] * torch.exp(4 * nbox[start + 4] - 2)
                    x = alpha_x + i * box_len
                    y = alpha_y + j * box_len

                    x = int(x.item())
                    y = int(y.item())
                    w = int(w.item())
                    h = int(h.item())

                    start = (int(x - w / 2), int(y - h / 2))
                    end = (int(x + w / 2), int(y + h / 2))
                    color = (255, 0, 0)
                    cv2.rectangle(image, start, end, color, 2)


def visualize():
    img, data = data_v.load_val()
    img_tensor = torch.from_numpy(img).float().cuda()
    data = torch.from_numpy(data).float().cuda()

    loss, y1, y2, y3 = model(img_tensor, data)

    # draw image
    img[0] = img[0] * 255.0
    image = img[0].transpose(2, 1, 0).astype("uint8")
    image = cv2.UMat(image).get()

    print_box(image, y1, img_size, 6)
    print_box(image, y2, img_size, 3)
    print_box(image, y3, img_size, 0)

    cv2.imshow("img", image)
    cv2.waitKey(0)


def main():
    if model.load(name="yolo_min_model"):
        print("model load end!!")
        model.cuda()
    else:
        print("model load error!!")
        return

    for i in range(100):
        visualize()


if __name__ == "__main__":
    main()