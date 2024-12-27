import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    # M 为 (batch_size, height, width)张量
    M = torch.sum(mask, (1, 2)) # 计算mask在空间维度(高和宽)上的求和，计算每张图片中被掩码标记为有效的像素个数。 得到的M为(batch_size,)的张量，即一维张量
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)



def ssi_trim_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    residual = prediction - target

    U_M = int(0.8 * M[0].item())
    abs_residual = torch.abs(residual)


    b, _, h, w = residual.shape
    m = h * w
    u_m = int(0.8 * m)
    abs_residual = torch.abs(residual)
    flat_abs_residual = abs_residual.view(b, -1)
    # Get an index of sorted abs_residual
    _, sorted_idx = torch.sort(flat_abs_residual.detach(), dim=1)
    # Get the top 80% of the sorted abs_residual
    top_80_idx = sorted_idx[:, :u_m]
    top_80_abs_residual = torch.gather(flat_abs_residual, 1, top_80_idx)
    # Sum the top 80% of the sorted abs_residual
    sum_top_80_abs_residual = torch.sum(top_80_abs_residual, dim=1)
    # Divide by the number of pixels
    loss_per_batch = sum_top_80_abs_residual / (2 * m)
    # Average over the batch
    loss = torch.mean(loss_per_batch)
    return loss


class MAETrimLoss(nn.Module):
    def __init__(self,reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return ssi_trim_loss(prediction,target,mask, reduction=self.__reduction)

class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        # self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


# 该版本还是需要加入mask再计算吧
# def ssi_trim_loss(residual):
#     b, _, h, w = residual.shape
#     m = h * w
#     u_m = int(0.8 * m)
#     abs_residual = torch.abs(residual)
#     flat_abs_residual = abs_residual.view(b, -1)
#     # Get an index of sorted abs_residual
#     _, sorted_idx = torch.sort(flat_abs_residual.detach(), dim=1)
#     # Get the top 80% of the sorted abs_residual
#     top_80_idx = sorted_idx[:, :u_m]
#     top_80_abs_residual = torch.gather(flat_abs_residual, 1, top_80_idx)
#     # Sum the top 80% of the sorted abs_residual
#     sum_top_80_abs_residual = torch.sum(top_80_abs_residual, dim=1)
#     # Divide by the number of pixels
#     loss_per_batch = sum_top_80_abs_residual / (2 * m)
#     # Average over the batch
#     loss = torch.mean(loss_per_batch)
#     return loss


# def ssi_trim_loss(residual, mask):
#     # 获取输入形状信息
#     b, _, h, w = residual.shape
#     m = h * w  # 总像素数量

#     # 展平 residual 和 mask
#     abs_residual = torch.abs(residual)
#     flat_abs_residual = abs_residual.view(b, -1)
#     flat_mask = mask.view(b, -1).to(dtype=flat_abs_residual.dtype)

#     # 仅选择有效像素的残差
#     valid_residual = flat_abs_residual * flat_mask  # 仅有效像素
#     valid_residual = valid_residual.view(b, -1)  # 展平处理

#     # 有效像素数量
#     valid_pixel_count = torch.sum(flat_mask, dim=1)  # 每张图片的有效像素数量
#     u_m = (0.8 * valid_pixel_count).long()  # 每张图片选择的 80% 的有效像素

#     # 排序，仅对有效像素进行排序
#     sorted_abs_residual, sorted_idx = torch.sort(valid_residual, dim=1)

#     # 选择排在前 80% 的像素
#     top_80_idx = sorted_idx[:, :torch.max(u_m)]  # 选择前 80% 索引
#     top_80_abs_residual = torch.gather(flat_abs_residual, 1, top_80_idx)

#     # 遍历批次，处理每张图像的有效像素数量
#     sum_top_80_abs_residual = torch.stack([
#         torch.sum(top_80_abs_residual[i, :u_m[i]])  # 每张图像的损失
#         for i in range(b)
#     ])

#     # 计算损失：归一化到有效像素总数范围
#     loss_per_batch = sum_top_80_abs_residual / (2 * valid_pixel_count)
#     loss = torch.mean(loss_per_batch)  # 批量损失均值

#     return loss