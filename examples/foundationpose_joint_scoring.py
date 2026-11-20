"""Joint scoring with the public FoundationPose code: the call pattern.

FoundationPose scores the hypotheses of one mask in one call. To compare the
best poses of several masks of the same object in one image, score those best
poses together in one further call and compare the scores of that call.

This file is an illustration and is not run by the tests of this repository:
it needs FoundationPose (https://github.com/NVlabs/FoundationPose; written
against estimater.py at commit a1b694b), its weights and a GPU. It is not the
code of the study. Run it from a FoundationPose checkout.
"""

import torch

from Utils import bilateral_filter_depth, erode_depth  # from the FoundationPose checkout


def register_masks_and_score_jointly(est, K, rgb, depth, masks, iteration=5):
    """Register each mask, then score the best poses of all masks in one call.

    est    FoundationPose estimator that was built for one object
    masks  boolean masks of that object in one image, one per detection
    Returns (mask indices, best poses in the model frame, per-call scores, joint scores).
    Rank the masks by the joint scores, not by the per-call scores.
    """
    # register() filters the depth in this way before it refines and scores
    filtered = bilateral_filter_depth(erode_depth(depth, radius=2, device="cuda"), radius=2, device="cuda")
    kept, best, per_call = [], [], []
    for index, mask in enumerate(masks):
        if int(((torch.as_tensor(filtered, device="cuda") >= 0.001) & torch.as_tensor(mask > 0, device="cuda")).sum()) < 4:
            continue  # register() returns a placeholder pose without scores for such masks
        est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)
        kept.append(index)
        best.append(est.pose_last.clone())  # best pose of this call, in the frame of the centred mesh
        per_call.append(float(est.scores[0]))
    if not best:
        return [], [], [], []
    joint, _ = est.scorer.predict(
        mesh=est.mesh, rgb=rgb, depth=filtered, K=K,
        ob_in_cams=torch.stack(best).data.cpu().numpy(), normal_map=None,
        mesh_tensors=est.mesh_tensors, glctx=est.glctx, mesh_diameter=est.diameter, get_vis=False)
    to_model = est.get_tf_to_centered_mesh()
    poses = [(pose @ to_model).data.cpu().numpy() for pose in best]
    return kept, poses, per_call, [float(score) for score in torch.as_tensor(joint).reshape(-1)]
