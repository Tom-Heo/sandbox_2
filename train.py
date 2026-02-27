import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from core.net import SandNet
from core.heo import Heo
from core.palette import Palette
from data.dataset import DIV2KDataset
from utils import get_kst_logger, EMA, CheckpointManager, Visualizer


def get_args():
    parser = argparse.ArgumentParser(description="SandNet Training Pipeline")

    # [필수] 재시작 / 재개 완벽 분리
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--restart", action="store_true", help="처음부터 새롭게 학습을 시작합니다."
    )
    group.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="지정한 체크포인트부터 학습을 재개합니다.",
    )

    # [데이터] DIV2K 자동 다운로드 지원
    parser.add_argument(
        "--download",
        action="store_true",
        help="DIV2K 데이터셋이 없으면 자동으로 다운로드합니다.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="데이터셋 루트 폴더 경로"
    )

    # [하이퍼파라미터]
    parser.add_argument("--epochs", type=int, default=1000, help="총 학습 에폭")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="데이터 로더 워커 수"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="기본 학습률")

    return parser.parse_args()


def main():
    args = get_args()

    # 1. 시스템 및 로거 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_kst_logger("SandNet", log_dir="logs")
    logger.info(
        f"디바이스: {device} | 학습 모드: {'Resume' if args.resume else 'Restart'}"
    )

    # 2. 데이터셋 및 로더 준비 (DIV2K 다운로드 및 로드 캡슐화)
    logger.info("데이터셋 초기화 중...")
    dataset = DIV2KDataset(
        root_dir=args.data_dir, scale_factor=4, download=args.download
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 3. 모델, 손실 함수, 옵티마이저 초기화
    model = SandNet().to(device)
    criterion = Heo.HeoLoss().to(device)
    optimizer = Heo.Heopimizer(model, lr=args.lr)

    # [Color Space] 색공간 변환기 (학습 파라미터 없음)
    srgb2oklab = Palette.sRGBtoOklabP().to(device)
    oklab2srgb = Palette.OklabPtosRGB().to(device)

    # 4. 유틸리티 (EMA, 체크포인트, 시각화) 초기화
    EMA_DECAY = 0.999
    ema = EMA(model, decay=EMA_DECAY)
    ckpt_manager = CheckpointManager(save_dir="checkpoints", max_keep=5)
    visualizer = Visualizer(output_dir="outputs")

    # 5. 스케줄러 설정 (10 Epoch Step-wise Warm-up + Step-wise Exponential Decay)
    SCHEDULER_GAMMA = 0.999996
    warmup_epochs = 10
    steps_per_epoch = len(dataloader)
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Warm-up 구간: 0.0에서 1.0으로 선형 증가
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Decay 구간: 지정된 Gamma 값으로 매 Step마다 지수적 감소
            decay_steps = current_step - warmup_steps
            return SCHEDULER_GAMMA**decay_steps

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 6. Restart / Resume 분기 처리
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {args.resume}")
        start_epoch = ckpt_manager.load(args.resume, model, ema, optimizer, scheduler)
        logger.info(
            f"[{args.resume}] 로드 완료. Epoch {start_epoch}부터 학습을 재개합니다."
        )
    else:
        logger.info("새로운 모델 가중치로 학습을 시작합니다. (--restart)")

    # 7. 메인 학습 루프
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (in_srgb, gt_srgb) in enumerate(dataloader):
            # 텐서를 디바이스로 이동
            in_srgb = in_srgb.to(device, non_blocking=True)
            gt_srgb = gt_srgb.to(device, non_blocking=True)

            # [Color Space] sRGB -> OklabP 변환 (역전파 그래프에서 분리)
            with torch.no_grad():
                in_oklab = srgb2oklab(in_srgb)
                gt_oklab = srgb2oklab(gt_srgb)

            # [Forward]
            pred_oklab = model(in_oklab)

            # [Loss] FP32 환경 강제 유지 (AMP 사용 안 함)
            loss = criterion(pred_oklab, gt_oklab)

            # [Backward & Optimize] Gradient Clipping 사용 안 함
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # [Step-wise Updates] 매 배치마다 스케줄러와 EMA 업데이트
            scheduler.step()
            ema.update()

            epoch_loss += loss.item()

        # Epoch 통계 로깅
        avg_loss = epoch_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch:04d}/{args.epochs:04d}] Loss: {avg_loss:.6f} | LR: {current_lr:.8f}"
        )

        # 8. Epoch 종료 후 평가 및 시각화 (EMA 가중치 적용)
        # 마지막 배치의 데이터(in_srgb, gt_srgb, in_oklab)를 대표 이미지로 재사용합니다.
        ema.apply_shadow()
        model.eval()

        with torch.no_grad():
            # EMA가 적용된 모델로 추론
            hr_oklab = model(in_oklab)

            # [Color Space] 시각화를 위해 OklabP -> sRGB로 원상 복구
            hr_srgb = oklab2srgb(hr_oklab)

            # (LR | HR | GT) 순서로 병합하여 outputs/ 폴더에 저장
            visualizer.save_epoch_result(epoch, in_srgb, hr_srgb, gt_srgb)

        # 다음 Epoch 학습을 위해 원본 가중치로 복구
        ema.restore()
        model.train()

        # 9. 체크포인트 저장 (최근 5개 유지)
        saved_path = ckpt_manager.save(epoch + 1, model, ema, optimizer, scheduler)
        logger.info(f"Epoch {epoch} 체크포인트 저장 완료: {saved_path}")


if __name__ == "__main__":
    main()
