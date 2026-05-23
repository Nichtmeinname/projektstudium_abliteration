import warnings

import torch

warnings.filterwarnings('ignore')


# GPU-Check und Setup
def setup_device():
    """
    Check gpu availability and setup device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU VERFÜGBAR!")
    else:
        device = torch.device('cpu')
        print("GPU NICHT VERFÜGBAR - Verwende CPU")
    return device
