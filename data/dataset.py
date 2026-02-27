import os
import sys  # 진행률 출력을 위한 내장 모듈 추가
import glob
import cv2
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset

from core.augments import DegradationPipeline
from utils import get_kst_logger


class DIV2KDataset(Dataset):
    """[DIV2K Dataset]
    DIV2K 고해상도(HR) 이미지를 자동으로 다운로드 및 압축 해제하고,
    실시간 열화 파이프라인을 거쳐 (Degraded, Clean) sRGB 텐서 쌍을 반환합니다.
    """

    DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"

    def __init__(
        self, root_dir: str = "data", scale_factor: int = 4, download: bool = False
    ):
        super().__init__()
        self.root_dir = root_dir
        self.extract_dir = os.path.join(root_dir, "DIV2K_train_HR")
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

        if not os.path.exists(zip_path):
            self.logger.info(f"DIV2K 데이터셋 다운로드 시작: {self.DIV2K_URL}")

            # [진행률 시각화 Hook] 외부 패키지 없이 순수 내장 모듈로 구현
            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)

                    # 40칸짜리 텍스트 진행 바 생성 (예: [████████----------])
                    bar_length = 40
                    filled_length = int(bar_length * percent / 100)
                    bar = "█" * filled_length + "-" * (bar_length - filled_length)

                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)

                    # '\r'(Carriage Return)을 사용해 콘솔의 같은 줄을 계속 덮어씁니다.
                    sys.stdout.write(
                        f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
                    )
                    sys.stdout.flush()

            # urlretrieve에 reporthook을 연결하여 청크(Chunk)를 받을 때마다 호출되게 합니다.
            urllib.request.urlretrieve(self.DIV2K_URL, zip_path, reporthook=reporthook)

            # 다운로드가 끝나면 다음 로그 출력이 덮어씌워지지 않도록 강제 줄바꿈을 해줍니다.
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

        degraded_tensor, clean_tensor = self.pipeline(image)
        return degraded_tensor, clean_tensor
