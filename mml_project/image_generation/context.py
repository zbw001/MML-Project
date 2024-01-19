from dataclasses import dataclass,field
from typing import Union, Optional, Callable, Tuple, Any, Dict
import torch

@dataclass
class AttnOptimContext:
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    params: Optional[torch.Tensor] = None
    radius: Optional[float] = None
    epoch: int = 0
    info: Dict[str, Any] = field(default_factory=dict)