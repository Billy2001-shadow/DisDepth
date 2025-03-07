import torch
import numpy as np
import torch.nn as nn


class SSIL_Loss(nn.Module):
    """
    Scale and Shift Invariant Loss
    """
    def __init__(self, valid_threshold=1e-8, max_threshold=1e8):
        super(SSIL_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold

    def inverse(self, mat):
        bt, _, _ = mat.shape
        mat += 1e-7 * torch.eye(2, dtype=mat.dtype, device=mat.device)
        a = mat[:, 0, 0]
        b = mat[:, 0, 1]
        c = mat[:, 1, 0]
        d = mat[:, 1, 1]
        ad_bc = a * d - b * c
        out = torch.zeros_like(mat)
        mat = mat / ad_bc[:, None, None]
        out[:, 0, 0] = mat[:, 1, 1]
        out[:, 0, 1] = -mat[:, 0, 1]
        out[:, 1, 0] = -mat[:, 1, 0]
        out[:, 1, 1] = mat[:, 0, 0]
        return out

    def scale_pred_depth_mask(self, pred, gt, logger=None):
        b, c, h, w = pred.shape
        mask = (gt > self.valid_threshold)  & (gt < self.max_threshold)  # [b, c, h, w]
        mask_float = mask.to(dtype=pred.dtype, device=pred.device)
        pred_valid = pred * mask_float   # [b, c, h, w]
        ones_valid_shape = torch.ones_like(pred_valid) * mask_float  # [b, c, h, w]
        pred_valid_ones = torch.cat((pred_valid, ones_valid_shape), dim=1)  # [b, c+1, h, w]
        pred_valid_ones_reshape = pred_valid_ones.reshape((b, c + 1, -1))  # [b, c+1, h*w]

        A = torch.matmul(pred_valid_ones_reshape, pred_valid_ones_reshape.permute(0, 2, 1))  # [b, 2, 2]

        # print(A)
        #A_inverse = (A + 1e-7 * torch.eye(2, dtype=A.dtype, device=A.device)).inverse() # this may get identity matrix in some versions of Pytorch. If it occurs, add 'print(A)' before it can solve it
        A_inverse = self.inverse(A)
        gt_valid = gt * mask_float
        gt_reshape = gt_valid.reshape((b, c, -1))  # [b, c, h*w]
        B = torch.matmul(pred_valid_ones_reshape, gt_reshape.permute(0, 2, 1))  # [b, 2, 1]
        scale_shift = torch.matmul(A_inverse, B)  # [b, 2, 1]
        ones = torch.ones_like(pred)
        pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
        pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2), scale_shift)  # [b, h*w, 1]
        pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))

        return pred_scale_shift, mask

    def forward(self, pred, gt, logger=None):
        pred_scale, mask_valid = self.scale_pred_depth_mask(pred, gt)
        valid_pixs = torch.sum(mask_valid, (1, 2, 3))
        valid_batch = valid_pixs > 50
        diff = torch.abs(pred_scale * valid_batch[:, None, None, None] * mask_valid -
                         gt*valid_batch[:, None, None, None]*mask_valid)
        loss = torch.sum(diff) / (torch.sum(valid_batch[:, None, None, None] * mask_valid) + 1e-8)
        return loss




class VNL_Loss(nn.Module):
    def __init__(self, focal_x, focal_y, input_size,
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.00001, sample_ratio=0.15):
        super(VNL_Loss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int32)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int32)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int32)
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index)) #[]
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        B, C, H, W = gt_depth.shape
        input_size = (H, W)
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss