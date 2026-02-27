import os
import sys
import time
import glob
import cv2
import random  # Random Crop을 위한 모듈 추가
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset

from core.augments import DegradationPipeline
from utils import get_kst_logger


class DIV2KDataset(Dataset):
    """[DIV2K Dataset]
    DIV2K 고해상도(HR) 이미지를 자동으로 다운로드 및 압축 해제하고,
    지정된 크기(patch_size)로 Random Crop을 수행한 뒤,
    실시간 열화 파이프라인을 거쳐 (Degraded, Clean) sRGB 텐서 쌍을 반환합니다.
    """

    DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"

    def __init__(
        self,
        root_dir: str = "data",
        scale_factor: int = 1,
        patch_size: int = 256,
        download: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.extract_dir = os.path.join(root_dir, "DIV2K_train_HR")
        self.patch_size = patch_size
        self.logger = get_kst_logger("SandNet")

        if download:
            self._download_and_extract()

        self.image_paths = sorted(glob.glob(os.path.join(self.extract_dir, "*.png")))

        if not self.image_paths:
            raise RuntimeError(
                f"'{self.extract_dir}' 경로에 이미지가 없습니다. --download 옵션을 주어 실행하세요."
            )

        self.pipeline = DegradationPipeline(scale_factor=scale_factor)

    def _download_and_extract(self):
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, "DIV2K_train_HR.zip")

        if (
            os.path.exists(self.extract_dir)
            and len(os.listdir(self.extract_dir)) >= 800
        ):
            self.logger.info(
                "DIV2K 데이터셋이 이미 온전하게 존재합니다. 다운로드를 건너뜁니다."
            )
            return

        if os.path.exists(zip_path):
            if not zipfile.is_zipfile(zip_path):
                self.logger.warning(
                    "손상되거나 불완전한 zip 파일이 발견되었습니다. 삭제 후 재다운로드합니다."
                )
                os.remove(zip_path)

        if not os.path.exists(zip_path):
            self.logger.info(f"DIV2K 데이터셋 다운로드 시작: {self.DIV2K_URL}")

            opener = urllib.request.build_opener()
            opener.addheaders = [
                ("User-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            ]
            urllib.request.install_opener(opener)

            last_time = 0.0

            def reporthook(block_num, block_size, total_size):
                nonlocal last_time
                downloaded = block_num * block_size

                if total_size > 0:
                    current_time = time.time()
                    percent = min(100.0, downloaded * 100.0 / total_size)

                    if current_time - last_time >= 0.2 or percent >= 100.0:
                        bar_length = 40
                        filled_length = int(bar_length * percent / 100)
                        bar = "█" * filled_length + "-" * (bar_length - filled_length)

                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)

                        sys.stdout.write(
                            f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
                        )
                        sys.stdout.flush()
                        last_time = current_time

            urllib.request.urlretrieve(self.DIV2K_URL, zip_path, reporthook=reporthook)
            sys.stdout.write("\n")
            self.logger.info("다운로드 완료.")

        self.logger.info("DIV2K 데이터셋 압축 해제 중...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root_dir)
        self.logger.info("압축 해제 완료.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(
                f"이미지 디코딩 실패. 파일이 손상되었을 수 있습니다: {path}"
            )

        # [고품질 축소] 원본 해상도의 가로세로를 정확히 1/2로 줄입니다.
        # 축소 과정의 앨리어싱(Aliasing) 억제에 가장 탁월한 INTER_AREA를 사용합니다.
        h_orig, w_orig = image.shape[:2]
        image = cv2.resize(
            image, (w_orig // 2, h_orig // 2), interpolation=cv2.INTER_AREA
        )

        # 1. Random Crop을 통한 공간 차원(Spatial Dimensions) 통일
        h, w = image.shape[:2]  # 1/2로 축소된 새로운 해상도를 기준으로 삼습니다.

        # [사소한 결계] 만약 축소된 이미지가 patch_size보다 작아졌다면 강제 확대
        if h < self.patch_size or w < self.patch_size:
            image = cv2.resize(
                image,
                (max(w, self.patch_size), max(h, self.patch_size)),
                # 확대할 때는 INTER_AREA가 아닌 INTER_CUBIC이 품질의 근본입니다.
                interpolation=cv2.INTER_CUBIC,
            )
            h, w = image.shape[:2]

        # 무작위 좌표 추출
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        # NumPy 메모리 슬라이싱
        cropped_image = image[y : y + self.patch_size, x : x + self.patch_size]

        # 2. 잘라낸 패치를 열화 파이프라인에 통과
        degraded_tensor, clean_tensor = self.pipeline(cropped_image)

        return degraded_tensor, clean_tensor
