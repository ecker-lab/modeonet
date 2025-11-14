from typing import Optional
import torch
from torch import nn
import numpy as np
from tensordict import tensorclass

from vibromodes.kirchhoff import PlateParameter
from vibromodes.utils import element_size
from scipy.signal import find_peaks





def complex_log_sum_dir(log_magnitude,dir_real,dir_imag,dim=0,keep_dim=False,epsilon=1e-6,stack_dim=-1):
    """
    log_magnitude: tensor
    dir: tensor
        dir_real**2 + dir_imag**2 = 1

    returns:
        log_magnitude
        normalized direction vector
        

    this calculates sum_i e^(amplitude_i + j phase)

    """
    max_magnitude = torch.max(log_magnitude.detach(),dim=dim,keepdim=True).values.detach()

    magnitude = torch.exp(log_magnitude-max_magnitude)
    real_number = (magnitude*dir_real).sum(dim=dim,keepdim=True)
    imag_number = (magnitude*dir_imag).sum(dim=dim,keepdim=True)


    epsilon = torch.finfo(real_number.dtype).eps
    sum_square = real_number.square() + imag_number.square() + epsilon
    result_ln_magnitude = 0.5*torch.log(sum_square)+max_magnitude
    result_direction = torch.stack([real_number,imag_number],dim=stack_dim)

    if not keep_dim:
        result_direction = result_direction.squeeze(dim)
        result_ln_magnitude = result_ln_magnitude.squeeze(dim)
    return result_ln_magnitude,result_direction

def calc_mode_response_complex(frequency,eigenfrequency2,loss_factor):
    """
    all components should have the same dimensions
    returns:
        mode_response
    """

    return frequency*1.j/ \
        ((1+1.j*loss_factor)*eigenfrequency2 - frequency**2)


def calc_mode_response(frequency,eigenfrequency2,loss_factor):
    """
    return 
        real_part
        imag_part
    """
    eps = torch.finfo(eigenfrequency2.dtype).eps
    real_part = loss_factor*frequency*eigenfrequency2 / \
        (loss_factor**2 * eigenfrequency2**2 + (frequency**2 - eigenfrequency2)**2 + eps)

    imag_part = (frequency*eigenfrequency2-frequency**3)/ \
        ((loss_factor**2 + 1 )* eigenfrequency2**2  
         + frequency**4 - 2*(frequency**2) * eigenfrequency2 + eps)
    
    return real_part,imag_part

def calc_mode_response_log_mag(frequency,eigenfrequency2,loss_factor):
    """
    all components should have the same dimensions
    returns:
        log_magnitude
        phase
    """


    loss_factor=loss_factor

    epsilon = torch.finfo(eigenfrequency2.dtype).eps
    log_mag = torch.log(frequency) \
        -0.5*torch.log(loss_factor**2 * eigenfrequency2**2 + (eigenfrequency2 - frequency**2)**2+epsilon)
    

    #scale with maximum
    #log_mag = log_mag - 0.5*torch.log(eigenfrequency2) + torch.log(loss_factor*eigenfrequency2)

    lower = eigenfrequency2-frequency**2

    phase = np.pi*0.5 - torch.arctan2(loss_factor*eigenfrequency2,lower)

    return log_mag,phase



