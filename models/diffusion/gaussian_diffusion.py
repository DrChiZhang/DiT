# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl, extract_into_tensor


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    根据名称获取预定义的 beta schedule（扩散过程中每一步添加的噪声强度表）。
    
    参数：
    - schedule_name: str，beta schedule 的名称（如 "linear", "squaredcos_cap_v2"）
    - num_diffusion_timesteps: int，总的扩散步骤数量 T

    返回：
    - 一个长度为 num_diffusion_timesteps 的 beta 数组，用于前向扩散过程。
    
    注：
    beta 是扩散模型中每一步添加的噪声方差，较小的 beta 表示保留更多原始图像的信息。
    """
    
    if schedule_name == "linear":
        # 如果选择线性beta调度器（来自 Ho et al. 的论文 DDPM）
        
        # 为了适配任意步数，将1000步标准调度器按比例缩放
        scale = 1000 / num_diffusion_timesteps
        
        # 返回线性变化的 beta 数组
        # beta 从 scale * 0.0001 线性增长到 scale * 0.02
        return get_beta_schedule(
            "linear",                             # 调度器类型
            beta_start=scale * 0.0001,            # 起始 beta 值（很小）
            beta_end=scale * 0.02,                # 终止 beta 值（较大）
            num_diffusion_timesteps=num_diffusion_timesteps  # 步数
        )

    elif schedule_name == "squaredcos_cap_v2":
        # 如果选择余弦调度器（来自 Improved-DDPM）
        
        # 使用 alpha_bar 函数（累计保留信息量）来间接计算 beta
        # 这个 lambda 函数定义了 alpha_bar 随时间 t 的变化曲线
        # t 是归一化时间（0~1），cos 余弦函数提供平滑下降效果
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            # 这个形式是 cap_v2 的关键，限制开头不要变化太快
        )

    else:
        # 如果传入了未知的调度器名称，则抛出错误提示
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")



def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior variance.
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])
        """
         Eq. from DDPM, Ho et al., Eq. 7
        """
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        For a given noisy image x at timestep t, apply the diffusion model to predict:
        - The mean and variance of the distribution for x_{t-1} (the next denoised step),
        - An estimate of the original clean image x_0.
        
        Parameters:
            model:          Neural network for denoising (takes x, t [, ...]).
            x:              Current noisy image tensor at timestep t, shape [B, C, ...].
            t:              1D tensor of current timesteps for each element in the batch, length B.
            clip_denoised:  Whether to clamp predicted x_start to [-1, 1] (pixel range).
            denoised_fn:    Optional function to further process the denoised image.
            model_kwargs:   Optional dictionary for extra model arguments (e.g. conditioning).
        
        Returns:
            dict with keys:
            - 'mean':         Model-predicted mean for p(x_{t-1} | x_t).
            - 'variance':     Model-predicted variance.
            - 'log_variance': Log of the model variance.
            - 'pred_xstart':  Model's prediction for x_0 (denoised clean image).
            - 'extra':        Any auxiliary output from the model (optional).
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        B, C = x.shape[:2]  # Batch size (B), number of channels (C)
        assert t.shape == (B,)

        # Feed input x at timestep t into the model
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            # Some models may output (prediction, extra)
            model_output, extra = model_output
        else:
            extra = None

        # Handle different model variance types
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 如果模型预测了方差（而不是使用固定方差），则 model_output 包含两部分：均值 和 编码后的方差
            # 断言检查 model_output 的形状是否符合预期
            # B: batch size, C: 通道数, x.shape[2:]: 空间维度，如图像为 (H, W)
            assert model_output.shape == (B, C * 2, *x.shape[2:])

            # 将 model_output 按通道维（dim=1）拆分成两个部分
            # 前一半为均值预测，后一半为方差相关值（通常是未归一化的 log-variance 编码）
            model_output, model_var_values = torch.split(model_output, C, dim=1)

            # 提取当前时间步 t 下，每个样本对应的最小 log-variance（来自 posterior 分布），
            # 并扩展到与 x 相同的形状（例如 [B, C, H, W]）
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

            # 提取当前时间步 t 下，每个样本对应的最大 log-variance，
            # 这里使用扩散过程中的 beta 值做参考（beta 越大，方差越大）
            max_log = extract_into_tensor(np.log(self.betas), t, x.shape)

            # 将模型预测的方差值 model_var_values 从 [-1, 1] 归一化到 [0, 1] 区间
            # 这是为了将其作为在 min_log 和 max_log 之间插值的比例 frac
            frac = (model_var_values + 1) / 2

            # 使用 frac 对最小和最大 log-variance 之间进行线性插值
            # frac 越大，model_log_variance 趋近于 max_log（预测方差越大）
            # frac 越小，model_log_variance 趋近于 min_log（预测方差越小）
            model_log_variance = frac * max_log + (1 - frac) * min_log

            # 将 log-variance 转换为真实的 variance 值，供采样使用
            model_variance = torch.exp(model_log_variance)
        else:
            # Fixed variance types (use precomputed/known values)
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)
        
        # Post-process x_start (predicted clean image) as needed
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Get x_start prediction
        if self.model_mean_type == ModelMeanType.START_X:
            # If model predicts x_0 directly
            pred_xstart = process_xstart(model_output)
        else:
            # If model predicts noise or velocity, recover x_0 via inverse formula
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        
        # Compute the mean of p(x_{t-1}|x_t, x_0) ("posterior mean")
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        # All outputs should have the same shape as the input x
        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        # Return all relevant values for the sampling step
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        P_theta x_(t-1) | x_t)
        Perform a single reverse diffusion step: sample x_{t-1} from x_t. 
        Parameters:
            model:          The neural network (diffusion model) used for prediction.
            x:              Current state/image at time t.
            t:              Current timestep, with 0 being the final step.
            clip_denoised:  Whether to clip x_start prediction to [-1, 1] (for images).
            denoised_fn:    Optional function for processing x_start before sampling.
            cond_fn:        Optional function for classifier-guided or other conditional sampling.
            model_kwargs:   Optional extra arguments for the model (e.g., conditioning).
        
        Returns:
            dict, with:
                - 'sample': the new sample (x_{t-1}) after this step,
                - 'pred_xstart': the predicted clean sample (x_0).
        """

        # Predict the posterior mean, variance, and model’s clean sample.
        """
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
            )
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # Generate Gaussian noise (of same shape as x) for stochastic sampling.
        noise = torch.randn_like(x)

        # Only add noise unless this is the last step (t == 0)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # Shape: (batch, 1, 1, 1, ...) for broadcasting

        # If using a conditional function (e.g., classifier-free guidance), adjust the mean.
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Sample from Gaussian: mean + std * noise, where std = exp(0.5 * log_variance)
        # No noise is added when t == 0 (final step).
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        # Return both the sample (for the next step) and the model's x_0 prediction.
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate the final sample(s) from the diffusion process by running the progressive sampling loop (step-by-step),
        but return only the last (fully denoised) result.
        
        Parameters:
            model:          the diffusion model module to use for sampling.
            shape:          desired output sample shape, e.g. (N, C, H, W).
            noise:          optional starting noise tensor (of the same shape), otherwise sampled randomly.
            clip_denoised:  whether to clip x_start predictions to [-1, 1].
            denoised_fn:    optional function to post-process denoised outputs before usage.
            cond_fn:        optional conditioning function, e.g. for classifier guidance.
            model_kwargs:   optional keyword arguments for conditional context/info.
            device:         torch device for allocating tensors, default is model's device.
            progress:       display a tqdm progress bar if True.
        
        Returns:
            The final ("fully denoised") output sample from the diffusion process.
        """
        final = None
        # Iterate through the progressive sampling generator.
        # Each step yields the sample's state at that diffusion step.
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            # Overwrite final at each step, so at end it's the last one.
            final = sample

        # Return only the ["sample"] key (the image tensor) from the final step.
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples
        at each timestep of the diffusion process, allowing step-wise observation.
        
        Parameters:
            model:          The diffusion model to use for generation.
            shape:          Shape of the generated data (e.g., (batch, channels, height, width)).
            noise:          Optional initial noise tensor. If None, sample from standard normal.
            clip_denoised:  Whether to clip the denoised samples (e.g., to [0, 1]).
            denoised_fn:    Optional function to modify denoised outputs.
            cond_fn:        Optional conditioning function for guided sampling.
            model_kwargs:   Additional arguments passed to the model.
            device:         Computation device (e.g., 'cuda' or 'cpu').
            progress:       Whether to display a progress bar.
        
        Returns:
            Generator yielding dictionaries, each corresponding to the output of p_sample() at a timestep.
        """
        # Automatically set device if not provided
        if device is None:
            device = next(model.parameters()).device

        # Ensure shape is a tuple or list
        assert isinstance(shape, (tuple, list))

        # Prepare the initial image tensor (noise or provided input)
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)  # Standard normal noise

        # Create a list of timestep indices in reverse order
        # 0, 1, 2, 3, 4 -> [0, 1, 2, 3, 4] -> [4, 3, 2, 1, 0]
        indices = list(range(self.num_timesteps))[::-1]

        # Wrap indices with a progress bar if requested
        if progress:
            # Lazy import to avoid unnecessary tqdm dependency if not used
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # Iteratively perform sampling from each timestep, yielding intermediate results
        for i in indices:
            # Create a tensor of current timestep for the batch, i=5, shape = (8, 3, 64, 64) (e.g., batch size 8), then [i] * shape[0] ⇒ [5, 5, 5, 5, 5, 5, 5, 5] (length 8)
            t = torch.tensor([i] * shape[0], device=device)

            with torch.no_grad():  # Disable gradients for faster inference
                # Call the sampling function for the current step
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out               # Yield intermediate result at this timestep
                img = out["sample"]     # Update the image for the next timestep

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                                                model=model,
                                                x_start=x_start,
                                                x_t=x_t,
                                                t=t,
                                                clip_denoised=False,
                                                model_kwargs=model_kwargs,
                            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [ ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                                                model=lambda *args, r=frozen_out: r, # Python匿名函数（lambda）+ 默认参数的典型组合用法，经常用来"临时伪装"一个模型接口，让其无论传入什么参数，都只返回你想要的那个张量frozen_out
                                                x_start=x_start,
                                                x_t=x_t,
                                                t=t,
                                                clip_denoised=False,
                            )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0  # Divide by 1000 for equivalence with initial implementation. Without a factor of 1/1000, the VB term hurts the MSE term.

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance( x_start=x_start, x_t=x_t, t=t )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
