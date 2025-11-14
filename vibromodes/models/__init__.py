
import numpy as np
import torch
from yacs.config import CfgNode as CN

from vibromodes.globals import AMPLITUDE_STD
from vibromodes.models.FQOUNet import FQO_Unet
from vibromodes.models.ModeONet import ModeONet

def default_model_config():
    config = CN()
    config.type = "InputIndependent"

    config.n_peaks = 20
    config.linear_rep = False

    config.compile = False
    config.temporal_dim = 128
    config.query_conditioning = "film"
    config.scaling_factor = 32
    config.analytic_response = False


    return config
    

def load_model(config):


    if config.model.type == "FQOUNet":
        model = FQO_Unet(conditional=True,
                         scaling_factor=32,
                         len_conditional=3,
                         output_size=config.mode_size)


    elif config.model.type == "ModeONet":
        model =  ModeONet(
            config.model.scaling_factor,config.model.n_peaks,
            config.mode_size,
            temporal_dim=config.model.temporal_dim,
            linear_prediction=config.model.linear_rep,
            query_conditioning_mode=config.model.query_conditioning,
            analytic_response=config.model.analytic_response,
        )
    else:
        raise NotImplementedError(f"Model type {config.model.type} is unkown.")
    
    """
    if config.precision==32:
        dtype = torch.float32
        torch.set_float32_matmul_precision('high')
    elif config.precision==64:
        dtype = torch.float64
    """
    if config.model.compile: 
        model.compile()

    return model.to()