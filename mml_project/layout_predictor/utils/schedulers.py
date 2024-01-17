from omegaconf import DictConfig
import torch
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

def build_optimizer_and_scheduler(optimize_cfg: DictConfig, hidden_size: int, max_steps: int, parameters) -> LRScheduler:
    scheduler_type = optimize_cfg.scheduler_type
    if scheduler_type == "BaseScheduler":
        initial_lr = hidden_size ** (- 0.5)
    elif scheduler_type == "ChrisScheduler" or scheduler_type == "BertScheduler":
        initial_lr = optimize_cfg.max_lr
    else:
        raise ValueError(f'Unsupported scheduler type: {scheduler_type}')

    if optimize_cfg.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=initial_lr, weight_decay=optimize_cfg.weight_decay)
    elif optimize_cfg.optimzier_type == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=initial_lr, weight_decay=optimize_cfg.weight_decay)

    lr_func = None
    if scheduler_type == 'BaseScheduler':
        warmup_steps = optimize_cfg.warmup_ratio * max_steps
        lr_func = lambda step : min(step ** (- 0.5), (warmup_steps ** (- 1.5))* step)
    elif scheduler_type == 'ChrisScheduler':
        warmup_steps = optimize_cfg.warmup_ratio * max_steps
        hold_steps = optimize_cfg.hold_ratio * max_steps
        decay_steps = optimize_cfg.decay_ratio * max_steps
        max_lr = optimize_cfg.max_lr
        min_lr = optimize_cfg.min_lr
        def lr_func(step):
            if step < warmup_steps: # linear
                lr = 0.1 * max_lr + (max_lr - 0.1 * max_lr) / warmup_steps * step
            elif step >= warmup_steps and step < hold_steps + warmup_steps: # hold
                lr = max_lr
            else: # 2 degree
                A = (max_lr * 0.9 / 1.0) ** 2. / decay_steps
                lr = -((step - (warmup_steps + hold_steps)) * A) ** 0.5 + max_lr
            if lr < min_lr:
                lr = min_lr
            return lr / initial_lr
    elif scheduler_type == 'BertScheduler':
        warmup_steps = optimize_cfg.warmup_ratio * max_steps
        hold_steps = optimize_cfg.hold_ratio * max_steps
        decay_steps = optimize_cfg.decay_ratio * max_steps
        max_lr = optimize_cfg.max_lr
        min_lr = optimize_cfg.min_lr
        def lr_func(step):
            if step < warmup_steps: # linear
                lr = 0.1 * max_lr + (max_lr - 0.1 * max_lr) / warmup_steps * step
            elif step >= warmup_steps and step < hold_steps + warmup_steps: # hold
                lr = max_lr
            else: # 2 degree
                A = max_lr / decay_steps
                lr = -((step - (warmup_steps + hold_steps)) * A) + max_lr
            if lr < min_lr:
                lr = min_lr
            return lr / initial_lr

    return optimizer, LambdaLR(optimizer, lr_func)