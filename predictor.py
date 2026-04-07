"""
钢索轴力神经网络预测模型
Cable Force Neural Network Predictor

支持6种已训练模型：
  - MLP (multilayer perceptron)
  - CNN (convolutional neural network)
  - RBFNN (radial basis function NN)
  - ModularRBFNN
  - PI-ModularRBFNN (physics-informed)
  - ProductNN

物理背景（来自 RWTH Aachen 实习论文）：
  输入1: 轴向位移 w ∈ [0, 0.025] m
  输入2: 温度     T ∈ [273, 1000] K
  输出:  钢索轴力  F (kN)
  归一化: min-max → [0, 1]，y_max ≈ 1155 kN

作者: 基于论文 draft28.tex 实现
"""

import os
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 归一化参数（与训练时完全一致）
# ─────────────────────────────────────────────────────────────────────────────
X_MIN = np.array([0.0,   273.0])   # [w_min (m), T_min (K)]
X_MAX = np.array([0.025, 1000.0])  # [w_max (m), T_max (K)]
Y_MIN = 0.0          # N (F_min = 0 when w=0)
Y_MAX = 1_155_281.4  # N (F_max ≈ 1155 kN at w=0.025m, T=273K)


# ─────────────────────────────────────────────────────────────────────────────
# 物理解析公式（后备 + 归一化用）
# ─────────────────────────────────────────────────────────────────────────────
_L   = 10.0          # 缆索长度 (m)
_r   = 0.03          # 截面半径 (m)
_A   = math.pi * _r**2  # 截面积 (m²)
_E   = 210e9         # 弹性模量 (Pa)
_k   = 252e6         # 屈服应力 (Pa)
_eps_p = 0.0012      # 屈服应变
_H0  = 120.64e9      # 塑性硬化模量基值 (Pa)
_B1  = 161e6         # 温度系数 (Pa/K)，修正后
_T0  = 1500.0        # 特征温度 (K)


def _H_modulus(T: float) -> float:
    """温度相关塑性硬化模量 H(T) = H0 - B1*T*exp(-T0/T)"""
    return _H0 - _B1 * T * math.exp(-_T0 / T)


def analytical_force(w: float, T: float) -> float:
    """
    双线性弹塑性本构模型 → 钢索轴力 (N)

    F = σ(ε, T) × A
    ε = w / L
    σ = E*ε         (ε ≤ ε_p)
      = k + H(T)*(ε - ε_p)  (ε > ε_p)
    """
    eps = w / _L
    if eps <= _eps_p:
        sigma = _E * eps
    else:
        sigma = _k + _H_modulus(T) * (eps - _eps_p)
    return sigma * _A  # N


# ─────────────────────────────────────────────────────────────────────────────
# 模型定义（与训练代码结构完全对应）
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """多层感知机 [256, 128, 64] + LeakyReLU + BatchNorm + Dropout"""
        def __init__(self, input_dim=2, output_dim=1,
                     hidden_dims=None, activation='leaky_relu',
                     negative_slope=0.01, use_batchnorm=True, dropout_rate=0.1):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [256, 128, 64]
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.LeakyReLU(negative_slope))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class CNN(nn.Module):
        """一维卷积网络 + 全局平均池化"""
        def __init__(self, input_dim=2, output_dim=1,
                     hidden_channels=None, kernel_sizes=None,
                     activation='relu', use_batchnorm=True, dropout_rate=0.1):
            super().__init__()
            if hidden_channels is None:
                hidden_channels = [32, 64, 128]
            if kernel_sizes is None:
                kernel_sizes = [3, 3, 3]
            self.input_dim = input_dim

            conv_blocks = []
            in_ch = 1
            for out_ch, ks in zip(hidden_channels, kernel_sizes):
                conv_blocks.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks//2))
                if use_batchnorm:
                    conv_blocks.append(nn.BatchNorm1d(out_ch))
                conv_blocks.append(nn.ReLU())
                if dropout_rate > 0:
                    conv_blocks.append(nn.Dropout(dropout_rate))
                in_ch = out_ch
            self.conv_blocks = nn.Sequential(*conv_blocks)

            # 全局平均池化后接全连接
            self.fc = nn.Sequential(
                nn.Linear(hidden_channels[-1], 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )

        def forward(self, x):
            # x: (batch, input_dim) → (batch, 1, input_dim)
            x = x.unsqueeze(1)
            x = self.conv_blocks(x)
            x = x.mean(dim=-1)  # 全局平均池化
            return self.fc(x)


    class RBFNN(nn.Module):
        """径向基函数网络，单层输出"""
        def __init__(self, input_dim=2, output_dim=1, num_centers=128,
                     sigma_trainable=True, init_method='grid',
                     min_sigma=0.05, max_sigma=1.0, output_activation=None):
            super().__init__()
            self.num_centers = num_centers
            self.min_sigma = min_sigma
            self.max_sigma = max_sigma

            if init_method == 'grid':
                side = int(math.sqrt(num_centers))
                g = torch.linspace(0, 1, side)
                gx, gy = torch.meshgrid(g, g, indexing='ij')
                centers = torch.stack([gx.flatten(), gy.flatten()], dim=1)
                # 如果网格点数与num_centers不完全匹配，用随机补齐或截断
                if centers.shape[0] > num_centers:
                    centers = centers[:num_centers]
                elif centers.shape[0] < num_centers:
                    extra = torch.rand(num_centers - centers.shape[0], input_dim)
                    centers = torch.cat([centers, extra], dim=0)
            else:
                centers = torch.rand(num_centers, input_dim)

            self.centers = nn.Parameter(centers)

            # sigma 通过 log_sigma_param 约束在 [min_sigma, max_sigma]
            init_sigma = (min_sigma + max_sigma) / 2
            init_logit = math.log(init_sigma - min_sigma + 1e-8)
            self.log_sigma_param = nn.Parameter(torch.full((num_centers,), init_logit))

            self.linear = nn.Linear(num_centers, output_dim)

        def _get_sigma(self):
            return self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(self.log_sigma_param)

        def forward(self, x):
            sigma = self._get_sigma()
            diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1)
            rbf_out = torch.exp(-dist_sq / (2 * sigma.unsqueeze(0) ** 2))
            return self.linear(rbf_out)


    class ModularRBFNN(nn.Module):
        """
        模块化RBFNN：每个center是(input_dim,)的完整向量，
        但mask只激活其对应的维度，实现维度解耦。

        checkpoint 中:
          centers:    (total_centers, input_dim)  — 每行一个center
          logit_sigma:(total_centers,)
          mask:       (total_centers, input_dim)  — 每行只有一个1
        """
        def __init__(self, input_dim=2, output_dim=1, num_centers_per_dim=128,
                     init_method='grid', use_bias=True,
                     min_sigma=0.05, max_sigma=0.5,
                     input_ranges=None,
                     output_type='mlp', hidden_sizes=None, dropout_rate=0.1):
            super().__init__()
            self.input_dim = input_dim
            self.num_centers_per_dim = num_centers_per_dim
            self.min_sigma = min_sigma
            self.max_sigma = max_sigma
            total_centers = input_dim * num_centers_per_dim

            # centers: (total_centers, input_dim)
            if init_method == 'grid':
                centers = torch.zeros(total_centers, input_dim)
                for d in range(input_dim):
                    g = torch.linspace(0, 1, num_centers_per_dim)
                    centers[d * num_centers_per_dim:(d + 1) * num_centers_per_dim, d] = g
            else:
                centers = torch.rand(total_centers, input_dim)
            self.centers = nn.Parameter(centers)

            self.logit_sigma = nn.Parameter(torch.zeros(total_centers))

            # mask: (total_centers, input_dim)，每行只有一个1
            mask = torch.zeros(total_centers, input_dim)
            for d in range(input_dim):
                mask[d * num_centers_per_dim:(d + 1) * num_centers_per_dim, d] = 1.0
            self.register_buffer('mask', mask)

            if hidden_sizes is None:
                hidden_sizes = [256, 128]

            layers = []
            prev = total_centers
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.LeakyReLU(0.01))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.output_layers = nn.Sequential(*layers)

        def forward(self, x):
            sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(self.logit_sigma)
            # x: (batch, input_dim)
            # 对每个center，只计算其激活维度上的距离
            # x_masked[b, c, d] = x[b, d] * mask[c, d]
            # c_masked[c, d]   = centers[c, d] * mask[c, d]
            x_m = x.unsqueeze(1) * self.mask.unsqueeze(0)   # (batch, total, input_dim)
            c_m = self.centers * self.mask                   # (total, input_dim)
            diff = x_m - c_m.unsqueeze(0)                   # (batch, total, input_dim)
            # 只对激活维度求距离（非激活维度 diff=0，贡献为0）
            dist_sq = (diff ** 2).sum(-1)                    # (batch, total)
            rbf_out = torch.exp(-dist_sq / (2 * sigma.unsqueeze(0) ** 2))
            return self.output_layers(rbf_out)


    class PIModularRBFNN(ModularRBFNN):
        """物理信息正则化ModularRBFNN，结构与ModularRBFNN完全相同"""
        pass


    class ProductNN(nn.Module):
        """乘积单元网络"""
        def __init__(self, input_dim=2, output_dim=1, hidden_dim=64,
                     activation='swish', use_batchnorm=True, dropout_rate=0.1):
            super().__init__()
            self.product_layer = nn.Linear(input_dim, hidden_dim)
            self.product_layer.bias_log = nn.Parameter(torch.zeros(hidden_dim))

            hidden_layers = []
            if use_batchnorm:
                hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            hidden_layers.append(nn.SiLU())  # Swish
            if use_batchnorm:
                hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden = nn.Sequential(*hidden_layers)
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # 乘积单元: exp(W * log(|x|+ε) + b)
            log_x = torch.log(torch.abs(x) + 1e-8)
            h = torch.exp(F.linear(log_x, self.product_layer.weight,
                                   self.product_layer.bias_log))
            h = self.hidden(h)
            return self.output_layer(h)


    MODEL_REGISTRY = {
        'mlp':              MLP,
        'cnn':              CNN,
        'rbf_nn':           RBFNN,
        'modularRBFNN':     ModularRBFNN,
        'pi_modularRBFNN':  PIModularRBFNN,
        'product_nn':       ProductNN,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 预测器主类
# ─────────────────────────────────────────────────────────────────────────────

class CableForcePredictor:
    """
    钢索轴力预测器

    自动扫描 models/ 目录，按优先级加载最佳模型（pi_modularRBFNN > mlp > ...）。
    若无模型文件，使用双线性本构解析公式作为后备。

    用法：
        predictor = CableForcePredictor()
        result = predictor.predict({"displacement": 0.012, "temperature": 636.0})
        print(result["predicted_force_kN"])
    """

    # 模型加载优先级（R²最高的优先）
    MODEL_PRIORITY = [
        'pi_modularRBFNN_seed42_best.pth',
        'mlp_seed42_best.pth',
        'modularRBFNN_seed42_best.pth',
        'rbf_nn_seed42_best.pth',
        'product_nn_seed42_best.pth',
        'cnn_seed42_best.pth',
        # 旧接口兼容
        'cable_rbfnn.pth',
    ]

    def __init__(self, models_dir: str = "models"):
        self.model = None
        self.model_name = None
        self.use_fallback = True
        self.models_dir = models_dir

        if not TORCH_AVAILABLE:
            print("ℹ️  PyTorch 未安装，使用解析公式后备")
            return

        # 按优先级尝试加载
        loaded = False
        for fname in self.MODEL_PRIORITY:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                loaded = self._load_model(path, fname)
                if loaded:
                    break

        if not loaded:
            print(f"ℹ️  models/ 目录下未找到模型文件，使用解析公式后备")
            print(f"   请将 .pth 文件放入 {models_dir}/ 目录")

    def _load_model(self, path: str, fname: str) -> bool:
        try:
            ckpt = torch.load(path, map_location="cpu")

            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                # 新格式：含 model_name 和 model_kwargs
                name = ckpt.get('model_name', '')
                kwargs = ckpt.get('model_kwargs', {})
                state_dict = ckpt['model_state_dict']

                model_cls = MODEL_REGISTRY.get(name)
                if model_cls is None:
                    print(f"⚠️  未知模型类型 {name}，跳过 {fname}")
                    return False

                model = model_cls(**kwargs)
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
                self.model_name = name
                self.use_fallback = False
                epoch = ckpt.get('epoch', '?')
                print(f"✅ 已加载模型: {fname}  (type={name}, epoch={epoch})")
                return True

            else:
                # 旧格式：直接 state_dict（兼容旧版 cable_rbfnn.pth）
                print(f"⚠️  {fname} 格式为旧版 state_dict，跳过")
                return False

        except Exception as e:
            print(f"⚠️  加载 {fname} 失败: {e}")
            return False

    # ── 推理 ────────────────────────────────────────────────────────────────

    def predict(self, params: dict) -> dict:
        """
        预测钢索轴力。

        参数 params 字典（支持两种输入风格）：
          新风格（与论文一致）：
            displacement (float): 轴向位移 w，单位 m，范围 [0, 0.025]
            temperature  (float): 温度 T，单位 K，范围 [273, 1000]

          旧风格（向后兼容）：
            temperature  (float): 温度，℃  → 自动转换为 K
            tension      (float): 张力，N   → 近似换算为位移
            elastic_modulus   (float): 弹性模量 GPa（可选，忽略）
            cross_section_area(float): 截面积 mm²（可选，忽略）

        返回字典：
            predicted_force_kN (float): 预测轴力 kN
            predicted_force_N  (float): 预测轴力 N
            lower_bound_kN     (float): 置信下界 kN
            upper_bound_kN     (float): 置信上界 kN
            method             (str):   使用的预测方法
            input_physical     (dict):  实际使用的物理输入 {w_m, T_K}
        """
        w, T_K = self._parse_inputs(params)

        if self.model is not None:
            pred_N, uncertainty_ratio = self._nn_predict(w, T_K)
            method = f"Neural Network ({self.model_name})"
        else:
            pred_N = analytical_force(w, T_K)
            uncertainty_ratio = 0.05
            method = "解析公式后备（双线性本构模型）"

        unc_N = abs(pred_N) * uncertainty_ratio
        pred_kN = pred_N / 1000.0

        return {
            "predicted_force_kN": pred_kN,
            "predicted_force_N":  pred_N,
            "predicted_force":    pred_kN,          # 兼容旧接口
            "lower_bound":        (pred_N - unc_N) / 1000.0,
            "upper_bound":        (pred_N + unc_N) / 1000.0,
            "lower_bound_kN":     (pred_N - unc_N) / 1000.0,
            "upper_bound_kN":     (pred_N + unc_N) / 1000.0,
            "method":             method,
            "input_physical":     {"w_m": w, "T_K": T_K},
            "input_params":       params,           # 兼容旧接口
        }

    def _parse_inputs(self, params: dict):
        """解析输入参数，统一转换为 (w_m, T_K)"""
        if "displacement" in params:
            # 新风格
            w   = float(params["displacement"])
            T_K = float(params.get("temperature", 636.0))
        else:
            # 旧风格兼容：temperature 以℃给出，tension 以 N 给出
            T_C = float(params.get("temperature", 20.0))
            T_K = T_C + 273.15

            tension_N = float(params.get("tension", 500.0))
            # F ≈ E * (w/L) * A  (弹性段) → w = F*L/(E*A)
            w = tension_N * _L / (_E * _A)
            w = max(0.0, min(w, 0.025))  # 夹在有效范围内

        # 夹入训练域
        w   = max(X_MIN[0], min(w,   X_MAX[0]))
        T_K = max(X_MIN[1], min(T_K, X_MAX[1]))
        return w, T_K

    def _nn_predict(self, w: float, T_K: float):
        """神经网络推理，返回 (force_N, uncertainty_ratio)"""
        import torch
        x_raw = np.array([[w, T_K]], dtype=np.float32)
        x_norm = (x_raw - X_MIN) / (X_MAX - X_MIN)  # [0,1] min-max
        x_t = torch.FloatTensor(x_norm)

        with torch.no_grad():
            y_norm = self.model(x_t).item()

        # 反归一化
        force_N = y_norm * (Y_MAX - Y_MIN) + Y_MIN
        uncertainty = 0.03  # 神经网络约3%不确定度
        return force_N, uncertainty

    # ── 便捷接口 ────────────────────────────────────────────────────────────

    def predict_from_wT(self, w_m: float, T_K: float) -> dict:
        """直接用物理单位预测"""
        return self.predict({"displacement": w_m, "temperature": T_K})

    def list_available_models(self) -> list:
        """列出 models/ 目录下可用的模型文件"""
        if not os.path.exists(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]


# ─────────────────────────────────────────────────────────────────────────────
# 快速自测
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("钢索轴力预测模型 - 自测")
    print("=" * 60)

    predictor = CableForcePredictor()

    # 用物理量直接测试
    test_cases = [
        {"displacement": 0.006,  "temperature": 273.0,  "desc": "弹性段，低温"},
        {"displacement": 0.012,  "temperature": 636.0,  "desc": "屈服点附近，中温"},
        {"displacement": 0.020,  "temperature": 800.0,  "desc": "塑性段，高温"},
        {"displacement": 0.025,  "temperature": 273.0,  "desc": "最大位移，低温（≈F_max）"},
    ]

    print(f"\n{'输入 w (m)':<12} {'T (K)':<8} {'预测 F (kN)':<14} "
          f"{'解析 F (kN)':<14} {'偏差%':<8} 描述")
    print("-" * 75)

    for case in test_cases:
        res = predictor.predict(case)
        analytical = analytical_force(case["displacement"], case["temperature"]) / 1000.0
        pred = res["predicted_force_kN"]
        err = abs(pred - analytical) / (analytical + 1e-6) * 100
        print(f"{case['displacement']:<12.3f} {case['temperature']:<8.0f} "
              f"{pred:<14.2f} {analytical:<14.2f} {err:<8.2f} {case['desc']}")

    print(f"\n方法: {res['method']}")
    print("=" * 60)
