import torch
import numpy as np
from vibromodes.globals import AMPLITUDE_STD, FRF_STD
from vibromodes.kirchhoff import tr_velocity_field_to_frequency_response
"""
VelocityField dict
{
    ln_mag : tensor with shape (...,H,W),
    normed_field: tensor with shape (...,2,H,W)
}
"""

def field_linear2dict(field):
    """
    vel_field shape: ... H x W
    
    result:
        VelocityField dict
    """
    
    amplitude = torch.square(torch.absolute(field))
    eps = torch.finfo(amplitude.dtype).eps
    ln_mag = torch.log(amplitude+1e-12)/AMPLITUDE_STD
    
    
    norm = torch.linalg.norm(field,dim=(-2,-1),keepdim=True)
    norm = torch.clip(norm,eps)

    if torch.is_complex(field):
        real = torch.real(field)
        imag = torch.imag(field)
        real = real/norm
        imag = imag/norm
        normalized_field = torch.stack([real,imag],dim=-3)
        return {
            "ln_mag": ln_mag,
            "normed_field": normalized_field
        }
    else:

        return {
            "ln_mag": ln_mag,
        }



def field_dict2linear(field_dict):
    """
    field_dict: VelocityField dict

    result:
        shape: ..., H x W  of dtype complex
    """

    normed = field_dict["normed_field"]
    real = normed[...,0,:,:]
    imag = normed[...,1,:,:]

    norm = torch.linalg.norm(normed,dim=-3)

    real = real/norm
    imag = imag /norm

    ln_mag = field_dict["ln_mag"]
    mag = (ln_mag*AMPLITUDE_STD*0.5).exp()

    return real*mag + 1.j*imag*mag


def field_dict2frf(field_dict,normalize=False):
    ln_mag = field_dict["ln_mag"]
    v_ref = 1e-9
    log_10_v_ref = -9
    ln2_vel_field = ln_mag*AMPLITUDE_STD

    B,F,W,H = ln2_vel_field.shape

    ln2_vel_field = ln2_vel_field.view(B,F,-1)
    #1./N for mean
    ln2_vel_field -= np.log(ln2_vel_field.shape[2])
    ln10_frf = torch.logsumexp(ln2_vel_field,dim=2)/np.log(10)
    v = 10*(ln10_frf-log_10_v_ref)
    if normalize:
        v/=FRF_STD


    return v

def field_linear2frf(field,normalize=False):
    return tr_velocity_field_to_frequency_response(field,normalization=normalize)

def linear2db(x):
    ln_db_ref = -9
    x = 10*(torch.abs(x).square().log10() - ln_db_ref)
    return x

def field_ln_mag_dir2dict(ln_mag_dir):
    """
    ln_mag_dir: 
        dict = {
            "ln_mag": tensor with shape (...,W,H)
            "dir": tensor with shape (...,2,W,H)
        }
    """
    dir = ln_mag_dir["dir"]
    ln_mag = ln_mag_dir["ln_mag"]
    eps = torch.finfo(dir.dtype).eps
    dir_norm = torch.clip(torch.linalg.norm(dir,dim=-3,keepdim=True),min=eps)
    direction = dir/dir_norm
    log_mag = ln_mag*AMPLITUDE_STD*0.5

    mag_shape = log_mag.shape
    log_mag = log_mag.flatten(start_dim=-2)


    mag = (log_mag - 0.5*torch.logsumexp(2*log_mag,dim=-1,keepdim=True)).exp()
    mag = mag.view(*mag_shape)
    normed_field = mag.unsqueeze(-3)*direction

    return {
        "ln_mag":ln_mag,
        "normed_field": normed_field
    }

def field_linear2ln_mag_dir(field):
    amplitude = torch.square(torch.absolute(field))+1e-12
    ln_mag = torch.log(amplitude)/AMPLITUDE_STD

    tmp = field/field.abs()
    dir_real = torch.real(tmp)
    dir_imag = torch.imag(tmp)

    dir = torch.stack([dir_real,dir_imag],dim=-3)

    return {
        "ln_mag": ln_mag,
        "dir":dir,
    }
    

