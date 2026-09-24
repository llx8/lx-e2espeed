import cupy as cp
import numpy as np
import torch
from cucim.core.operations.morphology import distance_transform_edt


def _external_stream_for(device):
    pt_stream = torch.cuda.current_stream(device=device)
    return cp.cuda.ExternalStream(pt_stream.cuda_stream)


def distance_transform_edt_torch(binary_mask: torch.Tensor) -> torch.Tensor:
    if binary_mask.device.type != "cuda":
        raise ValueError("distance_transform_edt_torch expects a CUDA tensor.")
    if binary_mask.ndim != 2:
        raise ValueError(f"distance_transform_edt_torch expects a 2D tensor, got shape={tuple(binary_mask.shape)}")
    with _external_stream_for(binary_mask.device):
        mask = (binary_mask > 0).to(torch.uint8).contiguous()
        mask_cu = cp.from_dlpack(mask)
        dist_cu = distance_transform_edt(mask_cu)
    return torch.from_dlpack(dist_cu)


def distance_transform_edt_np(binary_mask: np.ndarray, device: str | torch.device) -> np.ndarray:
    mask_t = torch.from_numpy(np.ascontiguousarray(binary_mask)).to(device=device, dtype=torch.uint8)
    dist_t = distance_transform_edt_torch(mask_t)
    return dist_t.cpu().numpy()
