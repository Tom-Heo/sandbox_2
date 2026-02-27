import cv2
import numpy as np
import random
import torch
from typing import Tuple


class DegradationPipeline:
    """
    현실 세계의 열화(Real-World Degradation) 과정을 물리적으로 모사합니다.
    SandNet의 공간적 해상도 유지 원칙에 따라 동일 해상도의 (sRGB,[0, 1]) 텐서 쌍을 반환합니다.
    """

    def __init__(self, scale_factor: int = 1):
        self.scale = scale_factor

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.2, 2.0)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    def _apply_downsample(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        interp = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR])
        return cv2.resize(img, (w // self.scale, h // self.scale), interpolation=interp)

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(0.0, 15.0)
        noise = np.random.normal(0.0, sigma, img.shape)
        noisy_img = img.astype(np.float32) + noise
        return np.clip(noisy_img, 0.0, 255.0).astype(np.uint8)

    def _apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(60, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    def real_world_degrade(self, image: np.ndarray) -> np.ndarray:
        """
        물리적 렌즈 결함(Blur) -> 센서 한계(Downsample) -> 센서 노이즈(Noise) -> 통신 손실(JPEG)
        *주의: OpenCV의 JPEG 압축은 BGR 기반 YCbCr 변환을 전제로 하므로 반드시 BGR 상태에서 수행합니다.
        """
        img = image.copy()
        img = self._apply_blur(img)
        img = self._apply_downsample(img)
        img = self._apply_noise(img)
        img = self._apply_jpeg_compression(img)
        return img

    def __call__(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (H, W, 3) BGR 원본 이미지 (np.uint8)
        Returns:
            degraded_tensor: (3, H, W) sRGB Tensor,[0.0, 1.0]
            clean_tensor: (3, H, W) sRGB Tensor,[0.0, 1.0]
        """
        h, w = image.shape[:2]

        # 1. BGR 환경에서 열화 수행 (OpenCV 연산 최적화)
        degraded_img = self.real_world_degrade(image)

        # 2. SandNet 규격에 맞춘 공간적 해상도 복구 (Bicubic)
        degraded_img = cv2.resize(degraded_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # 3. BGR -> RGB 색공간 변환
        clean_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)

        # 4. NumPy (H, W, C) -> PyTorch (C, H, W) 및 정규화 [0, 1]
        # div_()를 사용하여 In-place 연산으로 메모리 할당을 최소화합니다.
        clean_tensor = torch.from_numpy(clean_rgb).permute(2, 0, 1).float().div_(255.0)
        degraded_tensor = (
            torch.from_numpy(degraded_rgb).permute(2, 0, 1).float().div_(255.0)
        )

        return degraded_tensor, clean_tensor
