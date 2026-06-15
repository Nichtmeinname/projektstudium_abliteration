import gc

import torch


def modify_tensor_norm_preserved(
        W: torch.Tensor, direction: torch.Tensor, device, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating intervention direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype

    with torch.no_grad():
        # Move tensors for computation
        W_gpu = W.to(device, dtype=torch.float64, non_blocking=True)
        W_rank = W.dim()
        direction_gpu = direction.to(device, dtype=torch.float64, non_blocking=True)

        # Normalize intervention direction
        direction_normalized = torch.nn.functional.normalize(direction_gpu, dim=0)

        del direction_gpu  # cleanup

        # Case A: Standard Linear [Out, In] -> Transpose to [In, Out]
        if W_rank == 2:
            W_working = W_gpu.T
        elif W_rank == 3:
            W_working = W_gpu.permute(0, 2, 1)
        else:
            raise ValueError(f"Warning: Unsupported tensor shape {W_gpu.shape} - Skipping ablation.")

        del W_gpu  # cleanup

        # Decompose weight matrix
        # W_working is [in_features, out_features] or [Experts, in_features, out_features]
        W_norm = torch.norm(W_working, dim=-1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_working, dim=-1)  # normalized per output neuron

        del W_working  # cleanup

        # Apply ablation to the DIRECTIONAL component
        # Compute dot product of each row with intervention direction
        projection = torch.matmul(W_direction, direction_normalized)  # [in_features]

        # Subtract the projection
        W_direction_new = W_direction - scale_factor * (projection.unsqueeze(-1) * direction_normalized)

        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=-1)
        # Double-tap re-normalization — second pass catches residual from near-cancellation
        W_direction_new = W_direction_new - (W_direction_new @ direction_normalized).unsqueeze(
            -1) * direction_normalized
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=-1)

        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new

        # Transpose here to return safetensors convention
        if W_rank == 2:
            result = W_modified.T
        elif W_rank == 3:
            result = W_modified.permute(0, 2, 1)

        # Convert back to original dtype and CPU
        result = result.to(device, dtype=original_dtype, non_blocking=True)

        # Cleanup
        del direction_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result.detach().clone()
