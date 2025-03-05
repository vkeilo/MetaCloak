from torchvision import transforms
import torch
from copy import deepcopy
import cv2
from tqdm import tqdm
from robust_facecloak.generic.data_utils import load_data
import numpy as np

def first_pca_component(X):
    """
    计算输入数据的第一主成分，支持样本数 < 特征数的情况
    
    参数:
        X (Tensor): 形状为 (n_samples, n_features) 的二维张量
    
    返回:
        Tensor: 第一主成分向量，形状为 (n_features,)
    """
    # 数据标准化：减去均值
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # 执行奇异值分解 (SVD)
    # full_matrices=True 确保返回完整的右奇异向量矩阵
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=True)
    
    # 第一主成分是右奇异矩阵的第一行（对应最大奇异值）
    return Vh[0, :]  # 形状为 (n_features,)

def get_class2target_v_a():
    target_imgs = load_data(args.instance_data_dir_for_adversarial)
    class_imgs = load_data(args.class_data_dir)
    target_imgs_trans = all_trans(target_imgs).to(args.device, dtype=args.weight_dtype)
    class_imgs_trans = all_trans(class_imgs).to(args.device, dtype=args.weight_dtype)
    batch_size = 20

    target_imgs_latens = vae.encode(target_imgs_trans).latent_dist.sample()
    # class_imgs_latens = vae.encode(class_imgs_trans).latent_dist.sample()
    class_imgs_latens = torch.tensor([]).to(args.device, dtype=args.weight_dtype)
    for i in range(0, len(class_imgs), batch_size):
        tmp_class_imgs_latens = vae.encode(class_imgs_trans[i:i+batch_size]).latent_dist.sample()
        class_imgs_latens = torch.cat((class_imgs_latens, tmp_class_imgs_latens), dim=0)
    random_imgs_latens = torch.randn([10000,4,64,64]).to(args.device)
    mean_target_imgs_latens = torch.mean(target_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_class_imgs_latens = torch.mean(class_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_random_imgs_latens = torch.mean(random_imgs_latens, dim=0, keepdim=True).squeeze()
    del random_imgs_latens

    def get_orthog(x,y):
        y_a = y / torch.norm(y)
        x2y_pjt = (torch.dot(y,x)/torch.norm(y))*y_a
        x2y_orthor = x - x2y_pjt
        return x2y_orthor

    rand2class = mean_class_imgs_latens - mean_random_imgs_latens
    rand2target = mean_target_imgs_latens - mean_random_imgs_latens
    class2target = mean_target_imgs_latens - mean_class_imgs_latens
    # rand2class = torch.tensor([2,2]).to(dtype=args.weight_dtype)
    # rand2target = torch.tensor([1,2]).to(dtype=args.weight_dtype)
    # class2target = torch.tensor([-1,0]).to(dtype=args.weight_dtype)


    rand2class_flat = rand2class.flatten()
    rand2target_flat = rand2target.flatten()
    class2target_flat = class2target.flatten()
    mean_random_imgs_latens_flat = mean_random_imgs_latens.flatten()

    rand2target_to_rand2class_orthog = get_orthog(rand2target_flat,rand2class_flat)
    rand2target_to_rand2class_orthog_a = rand2target_to_rand2class_orthog/torch.norm(rand2target_to_rand2class_orthog)
    return rand2target_to_rand2class_orthog_a

def decode_latents(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def latents2img(latents, vae):
    target = latents
    # target = rand2target
    target = target.unsqueeze(0)
    invers_v = decode_latents(target,vae)
    inverse_transform = transforms.Normalize([-0.5]*3, [1/(0.5*255)]*3)
    invers_v_rev = inverse_transform(torch.tensor(invers_v).permute(0, 3, 1, 2)).permute(0, 2,3,1).numpy()
    return invers_v_rev[0].astype(np.uint8)

def get_orthogonal_vector(v):
    assert v.dim() == 1, "输入必须是一维张量"
    assert not torch.allclose(v, torch.zeros_like(v)), "输入向量不能为零向量"
    
    u = torch.randn_like(v)  # 随机生成向量
    coeff = torch.dot(u, v) / torch.dot(v, v)  # 投影系数
    u_ortho = u - coeff * v  # 减去投影分量
    
    # 处理极端情况：若结果为零向量，则重新生成
    if torch.allclose(u_ortho, torch.zeros_like(u_ortho)):
        return get_orthogonal_vector(v)
    u_ortho = u_ortho/torch.norm(u_ortho)
    return u_ortho
def get_first_principal_component(X):
    """
    输入:
        X : Tensor of shape (200, 16000)
        200个样本，每个样本16000维特征
    
    输出:
        pc1 : Tensor of shape (1, 16000)
        第一个主成分向量
    """
    # 数据标准化：中心化
    X_centered = X - X.mean(dim=0, keepdim=True)  # 按列求均值
    
    # 奇异值分解 (SVD)
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # 第一主成分是右奇异矩阵的第一行
    pc1 = Vh[0, :]  # 保持输出形状为(1, 16000)
    return pc1
def get_orthogonal_unit_vector(a,b):
    v = find_orthogonal_unit_vector(a,b).squeeze()
    v = v/torch.norm(v)
    return v
def find_orthogonal_unit_vector(a: torch.Tensor, b: torch.Tensor, eps=1e-8, max_attempts=100) -> torch.Tensor:
    """
    输入:
        a : 1D Tensor (n,)
        b : 1D Tensor (n,)
        eps : 防止除零的小量
        max_attempts : 最大尝试次数
    
    输出:
        ortho_vec : 与a、b同时正交的单位向量 (1, n)
    """
    assert a.dim() == 1 and b.dim() == 1, "输入必须是1D张量"
    assert a.size(0) == b.size(0), "向量维度需一致"
    
    device = a.device
    n = a.size(0)
    
    # 处理特殊零向量情况
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    a_zero = a_norm < eps
    b_zero = b_norm < eps
    
    if a_zero and b_zero:
        # 生成任意单位向量
        return torch.eye(n, device=device)[0:1]
    elif a_zero:
        # 仅需与b正交
        return _ortho_single(b.unsqueeze(0), eps, device)
    elif b_zero:
        # 仅需与a正交
        return _ortho_single(a.unsqueeze(0), eps, device)
    
    # 主处理流程
    for _ in range(max_attempts):
        # 生成随机向量并双重投影
        rand_vec = torch.randn(n, device=device)
        
        # 投影到a
        a_coeff = torch.dot(rand_vec, a) / (a_norm**2 + eps)
        ortho_a = rand_vec - a_coeff * a
        
        # 投影到b
        b_coeff = torch.dot(ortho_a, b) / (b_norm**2 + eps)
        ortho_ab = ortho_a - b_coeff * b
        
        # 检查有效性
        ortho_norm = torch.norm(ortho_ab)
        if ortho_norm > eps:
            return (ortho_ab / ortho_norm).unsqueeze(0)
    
    # 极端情况处理：使用SVD法
    return _svd_fallback(a, b, device)

def _ortho_single(vec: torch.Tensor, eps: float, device: torch.device) -> torch.Tensor:
    """处理单向量正交情况"""
    for _ in range(3):
        rand_vec = torch.randn(vec.size(1), device=device)
        proj = torch.dot(rand_vec, vec[0]) / (torch.norm(vec[0])**2 + eps)
        ortho = rand_vec - proj * vec[0]
        ortho_norm = torch.norm(ortho)
        if ortho_norm > eps:
            return (ortho / ortho_norm).unsqueeze(0)
    return torch.eye(vec.size(1), device=device)[1:2]

def _svd_fallback(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> torch.Tensor:
    """SVD降级方案"""
    # 构建约束矩阵
    constraints = torch.stack([a, b])  # (2, n)
    
    # 计算零空间基
    _, _, vh = torch.linalg.svd(constraints, full_matrices=True)
    null_space = vh[2:]  # 取第三行及之后
    
    # 返回第一个有效向量
    ortho_vec = null_space[0] / torch.norm(null_space[0])
    return ortho_vec.unsqueeze(0)

def get_some_principal_component(X,num):
    """
    输入:
        X : Tensor of shape (200, 16000)
        200个样本，每个样本16000维特征
    
    输出:
        pc1 : Tensor of shape (1, 16000)
        第一个主成分向量
    """
    # 数据标准化：中心化
    X_centered = X - X.mean(dim=0, keepdim=True)  # 按列求均值
    
    # 奇异值分解 (SVD)
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # 第一主成分是右奇异矩阵的第一行
    pc1 = Vh[:num, :]  # 保持输出形状为(1, 16000)
    return pc1

def find_orthogonal_unit_vector(vectors: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    输入:
        vectors : 包含n个1D张量的列表，每个张量形状为(dim,)
        eps : 数值稳定性阈值
    
    输出:
        ortho_vec : 与所有输入向量正交的单位向量 (1, dim)
    
    异常:
        当不存在非零正交向量时抛出ValueError
    """
    # 输入校验
    assert all(v.dim() == 1 for v in vectors), "所有输入必须为1D张量"
    dim = vectors[0].size(0)
    assert all(v.size(0) == dim for v in vectors), "所有向量维度必须一致"
    
    device = vectors[0].device
    n = len(vectors)
    
    # 基础情况处理
    if n == 0:
        return torch.eye(1, dim, device=device)  # 返回任意单位向量
    if dim <= n:
        raise ValueError(f"在{dim}维空间中无法找到与{n}个向量正交的非零向量")

    # 构造约束矩阵（自动处理线性相关性）
    A = vectors  # (n, dim)
    
    # 奇异值分解
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    
    # 确定有效秩（自动处理数值精度）
    rank = torch.sum(S > eps).item()
    
    if rank >= dim:
        raise ValueError("不存在正交非零解")
    
    # 提取零空间基向量
    null_space = Vh[rank:]  # (dim - rank, dim)
    
    # 选择第一个非零基向量
    for vec in null_space:
        vec_norm = torch.norm(vec)
        if vec_norm > eps:
            result = (vec / vec_norm).squeeze(0)  
            result = result/torch.norm(result)
            return result
    
    raise ValueError("无法找到有效正交向量")

def get_face_prob(image_path, net, confidence_threshold=0.5):
    # 读取图片
    image = cv2.imread(image_path)
    # print(image.shape)
    # print(image)
    h, w = image.shape[:2]
    
    # 预处理：调整大小并归一化
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    
    # 输入网络进行推断
    net.setInput(blob)
    detections = net.forward()
    
    # 解析检测结果，取最高置信度作为人脸概率
    max_confidence = 0.0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
    
    # 返回最高置信度（若大于阈值则认为是人脸）
    return max_confidence 

def get_face_prob_bylatens(latents, vae, net):
    # 读取图片
    image = latents2img(latents,vae)
    # print(image.shape)
    # print(image)
    h, w = image.shape[:2]
    
    # 预处理：调整大小并归一化
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    
    # 输入网络进行推断
    net.setInput(blob)
    detections = net.forward()
    
    # 解析检测结果，取最高置信度作为人脸概率
    max_confidence = 0.0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
    
    # 返回最高置信度（若大于阈值则认为是人脸）
    return max_confidence 

def get_largest_indices(lst, x):
    # 将每个元素和它的索引组合成一个元组列表
    indexed_lst = [(value, index) for index, value in enumerate(lst)]
    
    # 根据元素值降序排序
    indexed_lst.sort(reverse=True, key=lambda item: item[0])
    
    # 获取前x个元素的索引
    largest_indices = [indexed_lst[i][1] for i in range(x)]
    for i in largest_indices:
        print(lst[i])
    return largest_indices


def get_identify_feature_latents(args,vae,trans, max_feature = 50,select_feature = 5):
    vae = vae.to(args.device,dtype=args.weight_dtype)
    target_imgs = load_data(args.instance_data_dir_for_adversarial)
    class_imgs = load_data(args.class_data_dir)
    target_imgs_trans = trans(target_imgs).to(args.device, dtype=args.weight_dtype)
    class_imgs_trans = trans(class_imgs).to(args.device, dtype=args.weight_dtype)
    target_imgs_latens = vae.encode(target_imgs_trans).latent_dist.sample()
    class_imgs_latens = torch.tensor([]).to(args.device, dtype=args.weight_dtype)
    batch_size = 20
    print("start encode class imgs")
    for i in tqdm(range(0, len(class_imgs), batch_size)):
        tmp_class_imgs_latens = vae.encode(class_imgs_trans[i:i+batch_size]).latent_dist.sample()
        class_imgs_latens = torch.cat((class_imgs_latens, tmp_class_imgs_latens), dim=0)
    mean_target_imgs_latens = torch.mean(target_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_class_imgs_latens = torch.mean(class_imgs_latens, dim=0, keepdim=True).squeeze()

    class_imgs_latens_flat =  class_imgs_latens.flatten(1)
    # class_principal_component = get_some_principal_component(class_imgs_latens_flat,max_feature)
    # 特征还是从class2target的方向中提取更加合理
    class_principal_component = get_some_principal_component(class_imgs_latens_flat-mean_target_imgs_latens.flatten(),max_feature)
    class2target = mean_target_imgs_latens - mean_class_imgs_latens

    model_weights = "/data/home/yekai/github/sampleall/data/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model_config = "/data/home/yekai/github/sampleall/data/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(model_config, model_weights)
    
    face_rate_list = []
    for feature_id in range(max_feature):
        # start = 10
        # add_vnum = 1
        class_principal_component_mean = torch.mean(class_principal_component[feature_id:feature_id+1],dim=0)

        # tmp_v = deepcopy(rand2class)
        add_v = deepcopy(class_principal_component_mean)
        add_v = add_v/torch.norm(add_v)
        tmp_v = deepcopy(class2target)
        # steps = 10
        step_size = 1000
        tmp_v += add_v.reshape(4,64,64)*step_size
        print(feature_id)
        face_rate = get_face_prob_bylatens(tmp_v, vae, net)
        face_rate_list.append(face_rate)
    some_related_features = get_largest_indices(face_rate_list,select_feature)
    identify_face_feature = torch.mean(class_principal_component[some_related_features],dim=0)
    return identify_face_feature/torch.norm(identify_face_feature)