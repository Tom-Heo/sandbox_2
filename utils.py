import os
import glob
import logging
import torch
from datetime import datetime, timezone, timedelta
from torchvision.utils import save_image


def get_kst_logger(name: str = "SandNet", log_dir: str = "logs") -> logging.Logger:
    """
    [KST Logger]
    시스템 타임존에 의존하지 않고, 항상 '한국 표준시(UTC+9)' 기준으로 로그를 기록합니다.
    콘솔과 파일에 동시에 출력됩니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)

    # 중복 핸들러 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.propagate = False

    kst = timezone(timedelta(hours=9))

    class KSTFormatter(logging.Formatter):
        def converter(self, timestamp):
            return datetime.fromtimestamp(timestamp, tz=kst).timetuple()

    formatter = KSTFormatter(
        fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file = os.path.join(
        log_dir, f"train_{datetime.now(kst).strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class EMA:
    """
    [Exponential Moving Average]
    모델 파라미터의 이동 평균을 유지합니다.
    Step마다 아주 부드럽게(decay=0.999) 파라미터를 업데이트하며, 평가 시 원본 파라미터와 교체(Swap)합니다.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self):
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                # Integer 타입 버퍼(ex. num_batches_tracked)는 제외하고 소수점 파라미터만 업데이트
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """평가(Eval) 시작 전: 원본 파라미터를 백업하고 EMA 파라미터를 덮어씁니다."""
        self.backup = {
            k: v.clone().detach() for k, v in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.shadow)

    def restore(self):
        """평가(Eval) 종료 후: 백업해둔 원본 파라미터로 복구하여 학습을 이어갑니다."""
        self.model.load_state_dict(self.backup)
        self.backup = {}


class CheckpointManager:
    """
    [Checkpoint Manager]
    가장 최근 N개의 체크포인트만 디스크에 유지하여 용량 낭비를 막습니다.
    Resume(재개) 시 필요한 모델, EMA, 옵티마이저, 스케줄러 상태를 한 번에 묶어 저장/불러옵니다.
    """

    def __init__(self, save_dir: str = "checkpoints", max_keep: int = 5):
        self.save_dir = save_dir
        self.max_keep = max_keep
        os.makedirs(self.save_dir, exist_ok=True)

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        ema: EMA,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):

        path = os.path.join(self.save_dir, f"ckpt_epoch_{epoch:04d}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.shadow,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
        self._cleanup()
        return path

    def load(
        self,
        path: str,
        model: torch.nn.Module,
        ema: EMA,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> int:

        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        ema.shadow = ckpt["ema_state"]
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])

        return ckpt["epoch"]

    def _cleanup(self):
        """가장 오래된 체크포인트 삭제 (이름 순 정렬 보장)"""
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, "ckpt_epoch_*.pt")))
        if len(ckpts) > self.max_keep:
            for old_ckpt in ckpts[: -self.max_keep]:
                os.remove(old_ckpt)


class Visualizer:
    """
    [Result Visualizer]
    에폭마다 (LR | HR | GT) 순서로 이미지를 나란히 이어 붙여 저장합니다.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_epoch_result(
        self, epoch: int, lr: torch.Tensor, hr: torch.Tensor, gt: torch.Tensor
    ):
        """
        입력 텐서는 (B, 3, H, W) 형태의 sRGB [0, 1] 텐서여야 합니다.
        배치의 첫 번째 이미지 하나만 추출하여 저장합니다.
        """
        # Batch 차원 분리 후, 값의 범위를 안전하게 [0, 1]로 제한
        lr_img = lr[0].cpu().clamp(0.0, 1.0)
        hr_img = hr[0].cpu().clamp(0.0, 1.0)
        gt_img = gt[0].cpu().clamp(0.0, 1.0)

        # 가로(Width, dim=2) 방향으로 텐서 결합 (결과: 3, H, W*3)
        grid = torch.cat([lr_img, hr_img, gt_img], dim=2)

        path = os.path.join(self.output_dir, f"epoch_{epoch:04d}.png")
        save_image(grid, path)
