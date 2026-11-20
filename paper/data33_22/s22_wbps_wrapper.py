"""Score fixed original-object poses with the unmodified WAPR 0.0.3 WBPS."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WBPSGroup:
    within_group: object
    between_group: object
    per_pose_quality: object
    winner_index: int
    public_group_score: float


def score_fixed_pose_group(estimator, poses_m, rgb, depth_m, mask, K, mesh, center,
                           diameter_m, *, observed_xyz=None, mesh_id=None):
    """Put all supplied poses in one attention group and use official crops.

    The mesh, center, and diameter must be prepared exactly as for registration.
    The caller provides poses in the original object coordinate frame in meters.
    """
    import torch
    from wapr import recipe
    from wapr.estimator import poses_original_to_centered
    from wapr.ogl import make_crop_pair

    runner = estimator.nets["wbps"]
    if runner.engine is not None or runner.module.training:
        raise ValueError("Use the official Torch WBPS module in evaluation mode.")

    poses = torch.as_tensor(poses_m, device=estimator.device, dtype=torch.float32)
    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if poses.ndim != 3 or tuple(poses.shape[1:]) != (4, 4) or not len(poses):
        raise ValueError("Expected one or more original-frame 4x4 poses.")

    with torch.no_grad():
        centered = poses_original_to_centered(poses, center)
        rendered, observed, _, _ = make_crop_pair(
            rgb, depth_m, mask, centered, K, mesh, diameter_m,
            crop_ratio=recipe.wbps_crop_ratio, use_mask=False, device=estimator.device,
            observed_xyz=observed_xyz,
            mesh_ids=[mesh_id] * len(poses) if mesh_id is not None else None,
        )
        # The public runner rejects L=1. Its unchanged Torch module accepts it.
        outputs = runner.module(rendered, observed, L=len(poses))
        within = outputs["within_group"][0].float()
        between = outputs["between_group"][0].float()
        quality = (100.0 - between) / 200.0
        return WBPSGroup(
            within_group=within.cpu().numpy(),
            between_group=between.cpu().numpy(),
            per_pose_quality=quality.cpu().numpy(),
            winner_index=int(within.argmax()),
            public_group_score=float((100.0 - between.max()) / 200.0),
        )
