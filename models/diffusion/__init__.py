# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion

"""
这段代码是扩散模型(Diffusion Model)采样/推理/变步长训练中的“步长重采样调度器”。
它允许你从原始的（如1000步）扩散序列里按指定规则选择一小部分关键步，用来加速采样或者分段训练，是DDIM/DPM等采样器背后的核心逻辑
"""
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


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
    # Here, choose MSE by default
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
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing), # 完整的、一步不漏的1000步步长来跑模型采样/推理，不作下采样、跳步或子集选取。
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            ( gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL ) if not learn_sigma else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
