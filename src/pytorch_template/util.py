from typing import Optional
import numpy as np

import torch



def set_random_seed(seed: Optional[int] = None, is_test: Optional[bool] = None) -> None:
    if seed is not None: 
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if is_test:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
