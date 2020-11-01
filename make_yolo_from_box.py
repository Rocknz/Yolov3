import torch
from torch.autograd import Variable


def make_yolo_from_box(self, x, n_box, n_index, s_index, e_index):
    # x = [iou, x_a, y_a, w, h]

    check = Variable(torch.zeros(x.shape, dtype=torch.bool), requires_grad=False).cuda()
    check[:, 0, :, :] = True
    check[:, (5 + self.C), :, :] = True
    check[:, 2 * (5 + self.C), :, :] = True

    if n_box.shape[0] != 0:
        val = s_index <= n_index[:]
        val = val * (n_index[:] <= e_index)
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

            res_w = self.anchor[index, 0] * torch.exp(
                4 * now_x[now_index * (5 + self.C) + 3, box_iter] - 2
            )
            res_h = self.anchor[index, 1] * torch.exp(
                4 * now_x[now_index * (5 + self.C) + 4, box_iter] - 2
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

            iou = torch.ones([box_size]).cuda()
            loss = loss + self.lambda_coord * self.bce(
                now_x[now_index * (5 + self.C), box_iter], iou[:]
            )
            loss = loss + self.bce(label, hot_enco)
            loss = loss + self.lambda_coord * self.bce(res_alpha_x, alpha_x)
            loss = loss + 10 * self.lambda_coord * self.bce(res_alpha_y, alpha_y)
            loss = loss + 10 * self.lambda_coord * self.mse(
                res_w / self.img_size, box[:, 4] / self.img_size
            )
            loss = loss + self.lambda_coord * self.mse(
                res_h / self.img_size, box[:, 5] / self.img_size
            )

    x[check] = 0

    return x