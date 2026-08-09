import gc

import torch

from code.methods.setup_device import cleanup_gpu


def modify_tensor_norm_preserved(
        W: torch.Tensor,
        direction: torch.Tensor,
        device,
        scale_factor: float = 1.0,
        input_abliteration: bool = False
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
       Strength of the intervention. 1.0 removes the complete projection.

   input_abliteration : bool
       If Attention Tensors like Q, K, V or MLP Tensors like UP, GATE should be abliterated.

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
            if input_abliteration:
                # W: [out_features, in_features]
                # Each row is a vector in input space.
                W_working = W_gpu
            else:
                # W.T: [in_features, out_features]
                # Each row is a vector in output space.
                W_working = W_gpu.T
        else:
            raise ValueError(f"Warning: Unsupported tensor shape {W_gpu.shape} - Skipping ablation.")

        del W_gpu  # cleanup

        # Save original vector norms
        W_norm = torch.norm(W_working, dim=-1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_working, dim=-1)  # normalized per output neuron

        del W_working  # cleanup

        # Apply ablation to the DIRECTIONAL component
        # Remove intervention direction
        projection = torch.matmul(W_direction, direction_normalized)

        # Subtract the projection
        W_direction_new = W_direction - scale_factor * (projection.unsqueeze(-1) * direction_normalized)
        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=-1)

        # Double-tap re-normalization — second pass catches residual from near-cancellation
        # Numerical cleanup for complete abliteration only
        if scale_factor == 1.0:
            residual_projection = (
                    W_direction_new @ direction_normalized
            )

            W_direction_new = (
                    W_direction_new
                    - residual_projection.unsqueeze(-1)
                    * direction_normalized
            )

            W_direction_new = torch.nn.functional.normalize(
                W_direction_new,
                dim=-1
            )

        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new

        # Return to PyTorch [out, in] convention
        if input_abliteration:
            result = W_modified
        else:
            result = W_modified.T

        result = result.to(
            device=device,
            dtype=original_dtype,
            non_blocking=True
        )

        # Cleanup
        del direction_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified

        gc.collect()
        cleanup_gpu()

    return result.detach().clone()


def modify_tensor_standard(
        W: torch.Tensor,
        direction: torch.Tensor,
        device,
        scale_factor: float = 1.0,
        input_abliteration: bool = False
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
       Strength of the intervention. 1.0 removes the complete projection.

   input_abliteration : bool
       If Attention Tensors like Q, K, V or MLP Tensors like UP, GATE should be abliterated.

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
        intervention_dir_gpu = direction.to(device, dtype=torch.float64, non_blocking=True)

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
            if input_abliteration:
                # [out, in] -> direction in input space
                W_working = W_gpu
            else:
                # [out, in] -> [in, out]
                # direction in output space
                W_working = W_gpu.T
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
        W_modified = W_working - (scale_factor * (projection.unsqueeze(-1) * intervention_normalized))

        # Transpose here to return safetensors convention
        if input_abliteration:
            result = W_modified
        else:
            result = W_modified.T

        # Convert back to original dtype and CPU
        result = result.to(device, dtype=original_dtype, non_blocking=True)

        # Cleanup
        del intervention_normalized, projection, W_working

        gc.collect()
        cleanup_gpu()

    return result.detach().clone()
