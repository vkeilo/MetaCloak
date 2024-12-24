import torch
from torch.nn import Module
from skimage.restoration import denoise_tv_chambolle

class DenoiseTVChambolleTransform(Module):
    def __init__(self, weight=0.1, eps=2e-4, n_iter_max=200, multichannel=False):
        """
        初始化去噪参数。
        Args:
            weight (float): 平滑程度的权重。
            eps (float): 停止条件，表示优化收敛的阈值。
            n_iter_max (int): 最大迭代次数。
            multichannel (bool): 是否为多通道图像。
        """
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.n_iter_max = n_iter_max
        self.multichannel = multichannel

    def forward(self, img):
        """
        对输入图像应用 TV 去噪。
        Args:
            img (Tensor): 输入图像，形状为 [C, H, W] 或 [H, W]。
        Returns:
            Tensor: 去噪后的图像。
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input image must be a PyTorch Tensor.")
        
        # 将 Tensor 转为 NumPy 数组
        img_np = img.detach().cpu().numpy()
        
        # 检查图像维度
        if len(img_np.shape) == 3:  # [C, H, W] 转为 [H, W, C]（skimage 需要）
            img_np = img_np.transpose(1, 2, 0)
            self.multichannel = True
        elif len(img_np.shape) == 2:  # [H, W]
            self.multichannel = False
        else:
            raise ValueError("Input image must have shape [C, H, W] or [H, W].")
        
        # 应用 denoise_tv_chambolle
        denoised_np = denoise_tv_chambolle(
            img_np,
            weight=self.weight,
            eps=self.eps,
            n_iter_max=self.n_iter_max,
            multichannel=self.multichannel
        )
        
        # 将结果转回 Tensor，并转换为原始形状
        if self.multichannel:  # [H, W, C] 转回 [C, H, W]
            denoised_np = denoised_np.transpose(2, 0, 1)
        denoised_tensor = torch.tensor(denoised_np, dtype=img.dtype, device=img.device)
        return denoised_tensor