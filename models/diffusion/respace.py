# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion

class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    This class enables "timestep skipping" for acceleration (i.e., using a 
    subset of the diffusion steps for training/inference, as in fast DDPM/DDIM).
    
    Args:
        use_timesteps: A collection (sequence or set) of timesteps from the
                       original full diffusion process to retain for this run.
        kwargs:        Remaining arguments for constructing the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []                          # Maps reduced process timesteps back to original indices
        self.original_num_steps = len(kwargs["betas"])

        # Build the base (full) diffusion process
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []

        # Build new betas corresponding to only the selected steps in use_timesteps
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                 # Compute new beta to maintain correct transition variance for each interval
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # Call parent class method, passing wrapped model that handles timestep mapping
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # Ensure wrapped model is used for training loss computation
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        # Apply user condition mean function with wrapped model
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        # Apply user condition score function with wrapped model
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        # If already wrapped, return as is; otherwise, wrap to handle timestep mapping.
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.original_num_steps
        )

    # Scaling is handled by the wrapper model, so return as-is.
    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """
        Wraps the underlying model so that any caller (e.g. SpacedDiffusion) can use
        a reduced/accelerated set of diffusion steps. This function remaps timestep
        indices from the reduced process to the original indices for the underlying model.
        
        Args:
            x:        The input (images/latents).
            ts:       Step indices (in reduced process, e.g. 0..24 if using 25 steps).
            **kwargs: Additional arguments passed through.

        Returns:
            The model's output when called with x and (remapped) timesteps.

        Example:
            # For ts = tensor([0, 1, 2]) and
            #    self.timestep_map = [0, 40, 80, ..., 960],  ts=2 mapped to original t=80.
        """
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # If wanted: rescale timesteps to e.g., range [0,1000] as per DDPM conventions
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
