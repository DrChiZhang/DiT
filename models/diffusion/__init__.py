# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    """
    Constructs a SpacedDiffusion object, configuring noise schedule, loss type,
    and sampling/training options for a diffusion model.

    Args:
        timestep_respacing:      List/int/string of step spacings for accelerated sampling (e.g., [25] for 25 steps).
        noise_schedule:          Noise schedule for beta values ("linear", "cosine", etc).
        use_kl:                  If True, use a KL divergence loss (used in some diffusion training).
        sigma_small:             If True, use small fixed variance for transitions.
        predict_xstart:          If True, model predicts x_0 (the clean image); otherwise, predicts noise epsilon.
        learn_sigma:             If True, the model learns the variance ("sigma"); otherwise, it's fixed.
        rescale_learned_sigmas:  If True, rescales learned sigmas (typically for advanced models).
        diffusion_steps:         Total number of diffusion steps (default 1000).

    Returns:
        SpacedDiffusion object for training or sampling.
    """
    # Get the beta (noise) schedule for all diffusion steps, according to the requested schedule type
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    # Choose loss type: KL, rescaled MSE, or standard MSE
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # Set the step schedule for training/sampling, if not provided, use all steps (no respacing)
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    # Construct and return configured SpacedDiffusion object
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
