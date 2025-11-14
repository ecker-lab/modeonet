
from typing import Optional
import numpy as np
import torch
from torch import nn
from yacs.config import CfgNode as CN

def default_loss_config():
    config = CN()
    weights = CN()
    weights.pixel_l2 = 0.
    weights.aux_sparse = 0.
    weights.phase = 0.
    config.weights = weights

    return config


class LossDict:
    def __init__(self,values={},weights={}):
        assert set(values.keys())==set(weights.keys())
        self.values = values
        self.weights = weights

    def __or__(self,other):
        return LossDict(
            self.values | other.values,
            self.weights | other.weights
        )
    
    def __ior__(self,other):
        self.values |= other.values
        self.weights |= other.weights
        return self


    def sum(self):

        result = None

        for key in self.values:
            if result is None:
                result = self.values[key]*self.weights[key]
            else:
                result += self.values[key]*self.weights[key]
        return result
    
    def get_values(self):
        return {key:float(self.values[key]) for key in self.values}


class Loss(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.weights = weights
    
    def forward(self,pred_field,tgt_field,aux_loss):
        """
        pred_field: VelocityField dict
        tgt_field: VelocityField dict
        """

        loss_mag = nn.functional.mse_loss(pred_field["ln_mag"],tgt_field["ln_mag"])

        cos_sim = (pred_field["normed_field"]*tgt_field["normed_field"]).sum(dim=(-3,-2,-1)).mean()
        loss_phase = 1-cos_sim 

        loss_values = {"pixel_l2": loss_mag,
                       "phase": loss_phase,
                       }
        loss_values |= aux_loss

        weights = {w: self.weights[w] for w in loss_values}

        return LossDict(
            loss_values,weights
        )





def load_loss(config):
    loss_config = config.loss

    weights = {
        "pixel_l2": loss_config.weights.pixel_l2,
        "phase": loss_config.weights.phase,
        "aux_sparse": loss_config.weights.aux_sparse
    }

    loss = Loss(weights) 
    if(config.precision == 32):
        return loss.float()
    else:
        return loss.double()
    
