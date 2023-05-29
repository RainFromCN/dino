import torch.nn as nn
import numpy as np


def cancel_gradients_last_layer(model: nn.Module):
    for name, param in model.named_parameters():
        if "last_layer" in name:
            param.grad = None


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                     warmup_epochs=0, start_warmup_value=0):
    """
    cosine scheduler可以用在很多场景, 比如
        1. learning rate需要根据时间逐渐减小, 并且如果设置warmup则效果会更好
        2. weight decay应该随着时间逐渐增大, 无需设置warmup
    
    Parameters
    ----------
    base_value : Float. 初始值

    final_value : Float. 最终值

    epochs : 共需要多少个epoch

    niter_per_ep : 每个epoch有多少个batch, 可以用`len(data_loader)`获取

    warmup_epochs : [Optional] 升温过程持续多少个epoch
    
    start_warmup_value : 升温过程中的起点温度

    Returns
    -------
    schedule : 一个长度等于epoch * niter_per_ep的numpy数组
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = (final_value 
                + 0.5 * (base_value - final_value)
                * (1 + np.cos(np.pi * iters / len(iters)))
                )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def init_linear_module(module):
    assert isinstance(module, nn.Linear)
    nn.init.trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_layernorm_module(module):
    assert isinstance(module, nn.LayerNorm)
    nn.init.constant_(module.weight, 1)
    nn.init.constant_(module.bias, 0)


def init_clstoken(parameter):
    nn.init.trunc_normal_(parameter, std=0.02)


def init_posembed(parameter):
    nn.init.trunc_normal_(parameter, std=0.02)
