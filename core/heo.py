from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from collections import defaultdict
from math import sqrt
import math


class Heo:
    class HeLU(nn.Module):
        """
        원본 HeLU: last-dim 기반 (..., dim) 입력용
        """

        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 1.0))
            self.beta = nn.Parameter(torch.full((dim,), -1.0))
            self.redweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor):
            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            red = torch.tanh(sqrt(3.0) * self.redweight) + 1.0
            blue = torch.tanh(sqrt(3.0) * self.blueweight) + 1.0
            redx = rgx * red
            bluex = bgx * blue
            x = redx + bluex
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            y = (alpha * x + beta * raw) / 2
            return y

    class HeLU2d(nn.Module):
        """
        입력: (N,C,H,W)
        """

        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            # 원본 HeLU와 같은 파라미터 의미(채널별)
            self.alpha = nn.Parameter(torch.full((c,), 1.0))
            self.beta = nn.Parameter(torch.full((c,), -1.0))
            self.redweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise ValueError(
                    f"HeLU2d expects NCHW 4D tensor, got shape={tuple(x.shape)}"
                )
            if x.size(1) != self.channels:
                raise ValueError(
                    f"HeLU2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            # (C,) -> (1,C,1,1) broadcasting
            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, -1, 1, 1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, -1, 1, 1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            y = (alpha * x + beta * raw) / 2
            return y

    class Heopimizer(AdamW):
        """
        HeLU 및 HeLU2d의 파라미터가 0으로 수렴하지 않도록 Weight Decay를 해제하고,
        각 레이어의 채널 차원(dim)에 정확히 비례하여 학습률(lr * dim)을 부여하는 맞춤형 옵티마이저입니다.
        """

        def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
            **kwargs,
        ):
            base_params = []
            # dim(채널 수)을 키(key)로 삼아 파라미터들을 동적으로 그룹화합니다.
            helu_by_dim = defaultdict(list)

            # 1. 모델 내부 파라미터 순회 및 분류
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # HeLU 계열 파라미터 식별
                if any(
                    key in name for key in ("alpha", "beta", "redweight", "blueweight")
                ):
                    # 파라미터의 첫 번째 차원 크기가 곧 해당 레이어의 dim(channels)입니다.
                    dim = param.shape[0]
                    helu_by_dim[dim].append(param)
                else:
                    base_params.append(param)

            # 2. 기본 파라미터 그룹 생성
            param_groups = [
                {"params": base_params, "lr": lr, "weight_decay": weight_decay}
            ]

            # 3. HeLU 파라미터들을 차원(dim)별로 묶어 그룹 추가
            for dim, params in helu_by_dim.items():
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr * dim,  # 요청하신 대로 lr에 dim을 곱하여 증폭
                        "weight_decay": 0.0,  # Weight Decay 해제
                    }
                )

            # 4. 부모 클래스(AdamW) 초기화
            super().__init__(param_groups, **kwargs)

    class HeoGate(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.0))
            self.beta = nn.Parameter(torch.full((dim,), 0.0))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            return (alpha * x + beta * raw) / 2

    class HeoGate2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            self.alpha = nn.Parameter(torch.full((c,), 0.0))
            self.beta = nn.Parameter(torch.full((c,), 0.0))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            if x.dim() != 4 or x.size(1) != self.channels:
                raise ValueError(
                    f"HeoGate2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            return (alpha * x + beta * raw) / 2

    class HeoLoss(nn.Module):
        def __init__(self, epsilon=1 / (math.e - 1)):
            super().__init__()
            self.epsilon = epsilon
            self.epsilon_char = 1e-8

        def forward(self, pred, target):
            pred = pred.float()  # FP32 강제
            target = target.float()

            diff = pred - target
            abs_diff = diff.abs()

            charbonnier = torch.sqrt(diff**2 + self.epsilon_char**2)

            sharp_loss = torch.log(1 + 1000.0 * charbonnier / self.epsilon) / 1000.0

            l1_loss = diff.abs()

            loss = torch.where(abs_diff <= 0.001, sharp_loss, l1_loss)

            return loss.mean()
