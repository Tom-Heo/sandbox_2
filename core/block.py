import torch
import torch.nn as nn
import math
from .heo import Heo


class SimpleGate(nn.Module):
    """
    NAFNet의 SimpleGate 메커니즘.
    입력 텐서의 채널을 3개로 분할하여 요소별 곱셈(Element-wise multiplication)을 수행합니다.

    - Input shape: (B, C * 3, H, W)
    - Output shape: (B, C, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 채널 차원(dim=1)을 기준으로 텐서를 3개로 분할합니다.
        x1, x2, x3 = x.chunk(3, dim=1)

        # 세 텐서의 요소별 곱
        prod = x1 * x2 * x3

        # [사소한 공들임] 파이썬 내장 math.cbrt 대신 PyTorch 텐서 연산 사용
        # 미분 그래프(Autograd)를 끊지 않고, 음수 값의 세제곱근을 안전하게 계산합니다.
        return torch.sign(prod) * (torch.abs(prod) + (1e-8)).pow(1.0 / 3.0)


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.conv0001 = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.conv001 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0)
        self.act1 = Heo.HeLU2d(dim * 3)
        self.simple_gate = SimpleGate()
        self.conv01 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.residual_gate = Heo.HeoGate2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x
        x = self.conv0001(x)
        x = self.conv001(x)
        x = self.act1(x)
        x = self.simple_gate(x)
        x = self.conv01(x)
        x = self.residual_gate(x, raw)
        return x


class Stage(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.block1 = Block(dim)
        self.block2 = Block(dim)
        self.block3 = Block(dim)
        self.block4 = Block(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.block1 = Block(dim)
        self.block2 = Block(dim)
        self.block3 = Block(dim)
        self.block4 = Block(dim)
        self.block5 = Block(dim)
        self.block6 = Block(dim)
        self.block7 = Block(dim)
        self.block8 = Block(dim)
        self.block9 = Block(dim)
        self.block10 = Block(dim)
        self.block11 = Block(dim)
        self.block12 = Block(dim)
        self.block13 = Block(dim)
        self.block14 = Block(dim)
        self.block15 = Block(dim)
        self.block16 = Block(dim)
        self.block17 = Block(dim)
        self.block18 = Block(dim)
        self.block19 = Block(dim)
        self.block20 = Block(dim)
        self.block21 = Block(dim)
        self.block22 = Block(dim)
        self.block23 = Block(dim)
        self.block24 = Block(dim)
        self.block25 = Block(dim)
        self.block26 = Block(dim)
        self.block27 = Block(dim)
        self.block28 = Block(dim)
        self.block29 = Block(dim)
        self.block30 = Block(dim)
        self.block31 = Block(dim)
        self.block32 = Block(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.block24(x)
        x = self.block25(x)
        x = self.block26(x)
        x = self.block27(x)
        x = self.block28(x)
        x = self.block29(x)
        x = self.block30(x)
        x = self.block31(x)
        x = self.block32(x)

        return x
