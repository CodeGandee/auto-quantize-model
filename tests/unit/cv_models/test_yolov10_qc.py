from __future__ import annotations

import torch
from torch import nn

from auto_quantize_model.cv_models.yolov10_qc import QcBatchNorm2d, insert_qc_modules, run_qc_training


def test_qc_wrapper_only_qc_params_trainable() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(),
    )

    inserted = insert_qc_modules(model)
    assert inserted == 1
    assert isinstance(model[1], QcBatchNorm2d)

    batches = [{"img": torch.randn(2, 3, 8, 8)} for _ in range(2)]
    _ = run_qc_training(model=model, train_batches=batches, device=torch.device("cpu"), lr=1e-3, epochs=1)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable == ["1.m_gamma", "1.m_beta"]
    assert model[1].m_bn.training is False

