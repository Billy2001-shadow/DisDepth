import torch
import numpy as np

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


# RunningAverage 类 负责维护一个数值的实时平均值。每次调用 append 方法时，它会更新这个平均值，并返回当前的平均值。
class RunningAverage:
	def __init__(self):
		self.avg = 0
		self.count = 0

	def append(self, value):
		self.avg = (value + self.count * self.avg) / (self.count + 1)
		self.count += 1

	def get_value(self):
		return self.avg

class RunningAverageDict:
	def __init__(self):
		self._dict = None

	def update(self, new_dict):
		if self._dict is None:
			self._dict = dict()
			for key, value in new_dict.items():
				self._dict[key] = RunningAverage()

		for key, value in new_dict.items():
			self._dict[key].append(value)

	def get_value(self):
		return {key: value.get_value() for key, value in self._dict.items()}

# def compute_errors(gt, pred):
# 	assert pred.shape == gt.shape
	
# 	thresh = np.maximum((gt / pred), (pred / gt))
# 	d1 = (thresh < 1.25).mean()
# 	d2 = (thresh < 1.25 ** 2).mean()
# 	d3 = (thresh < 1.25 ** 3).mean()

# 	abs_rel = np.mean(np.abs(gt - pred) / gt)
# 	sq_rel = np.mean(((gt - pred) ** 2) / gt)

# 	rmse = (gt - pred) ** 2
# 	rmse = np.sqrt(rmse.mean())

# 	rmse_log = (np.log(gt) - np.log(pred)) ** 2
# 	rmse_log = np.sqrt(rmse_log.mean())

# 	err = np.log(pred) - np.log(gt)
# 	silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

# 	log10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
	

# 	return dict(d1=d1, d2=d2, d3=d3, abs_rel=abs_rel, rmse=rmse, log10=log10, rmse_log=rmse_log,
# 				silog=silog, sq_rel=sq_rel)


def compute_errors(gt, pred):
    assert pred.shape == gt.shape
    
    # 添加一个小的常数以避免对零值取对数
    epsilon = 1e-8
    gt = gt + epsilon
    pred = pred + epsilon
    
    # 过滤掉无效值
    valid_mask = (gt > epsilon) & (pred > epsilon)
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    
    return dict(d1=d1, d2=d2, d3=d3, abs_rel=abs_rel, rmse=rmse, log10=log10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred