import gc

import torch


def modify_tensor_norm_preserved(
        W: torch.Tensor,
        direction: torch.Tensor,
        device,
        scale_factor: float = 1.0
) -> torch.Tensor:
    """
    Norm preserving Abliteration.

    Modify weight tensor by ablating intervention direction while preserving row norms.

   Parameters
   ----------
   W : torch.Tensor
       Weight tensor of shape [Out, In] or [Experts, Out, In].

   direction : torch.Tensor
       Intervention direction.

   device :
       CUDA or CPU device.

   scale_factor : float
       Strength of the intervention.
       1.0 removes the complete projection.

   Returns
   -------
   torch.Tensor
       Modified weight tensor.
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


def modify_tensor_standard(
        W: torch.Tensor,
        direction: torch.Tensor,
        device,
        scale_factor: float = 1.0
) -> torch.Tensor:
    """
   Standard Abliteration.

   Removes the projection of each weight vector onto the intervention
   direction without preserving the original vector norm.

   Parameters
   ----------
   W : torch.Tensor
       Weight tensor of shape [Out, In] or [Experts, Out, In].

   direction : torch.Tensor
       Intervention direction.

   device :
       CUDA or CPU device.

   scale_factor : float
       Strength of the intervention.
       1.0 removes the complete projection.

   Returns
   -------
   torch.Tensor
       Modified weight tensor.
   """
    original_dtype = W.dtype

    with torch.no_grad():
        # Move tensors for computation
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True)
        W_rank = W.dim()
        intervention_dir_gpu = direction.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure intervention_dir is a 1-dimensional tensor
        if intervention_dir_gpu.dim() > 1:
            intervention_dir_gpu = intervention_dir_gpu.view(-1)

        # Normalize intervention direction
        intervention_normalized = torch.nn.functional.normalize(intervention_dir_gpu, dim=0)

        del intervention_dir_gpu  # cleanup

        # Transpose here to convert from safetensors convention
        # Handle Shapes: We want the "Output" dimension to be the last dimension for projection.
        # Intervention Vector lives in the Output Space.

        # Case A: Standard Linear [Out, In] -> Transpose to [In, Out]
        if W_rank == 2:
            W_working = W_gpu.T
        # Case B: Fused Experts [Experts, Out, In] -> Permute to [Experts, In, Out]
        # ex: GPT-OSS-20b
        elif W_rank == 3:
            W_working = W_gpu.permute(0, 2, 1)
        else:
            print(f"Warning: Unsupported tensor shape {W_gpu.shape} - Skipping ablation.")
            return W

        del W_gpu  # cleanup

        # Apply ablation
        # Compute dot product of each row with intervention direction
        # [..., Out] @ [Out] -> [...,]
        projection = torch.matmul(W_working, intervention_normalized)

        # Subtract the projection
        # [...,] -> [..., 1] * [Out] -> [..., Out]
        W_working -= scale_factor * (projection.unsqueeze(-1) * intervention_normalized)

        # Transpose here to return safetensors convention
        if W_rank == 2:
            result = W_working.T
        elif W_rank == 3:
            result = W_working.permute(0, 2, 1)

        # Convert back to original dtype and CPU
        result = result.to(device, dtype=original_dtype, non_blocking=True)

        # Cleanup
        del intervention_normalized, projection, W_working

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result.detach().clone()
