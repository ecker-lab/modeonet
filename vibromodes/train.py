import wandb
import torch
from vibromodes.hdf5_dataset import BatchData
from vibromodes.globals import AMPLITUDE_STD
from vibromodes.log import Logger
from vibromodes.loss import LossDict
from vibromodes.metrics import calc_metrics, evaluate_model
from vibromodes.kirchhoff import tr_velocity_field_to_frequency_response
import numpy as np

from torch.profiler import profile, ProfilerActivity, record_function
from time import time

from vibromodes.velocity_field import field_dict2frf

class StatsTracker:
    def __init__(self):
        self.stats = {}
        self.counter = {}

    def update(self,stats:dict,batch_size):

        for key in stats:
            if key not in self.stats:
                self.stats[key] = stats[key]*batch_size
                self.counter[key] = batch_size
            else:
                self.stats[key] += stats[key]*batch_size
                self.counter[key] += batch_size

            

    def finish(self)->dict:
        return {key: self.stats[key]/self.counter[key] for key in self.stats}


class SparseScheduler:
    def __init__(self,epochs,max_epoch,min_val,max_val):
        self.epochs = epochs
        self.max_epoch = max_epoch
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = min_val

    def step(self,epoch):
        if epoch < self.max_epoch:        
            factor = epoch/self.max_epoch
        else:
            factor = (epoch-self.max_epoch)/(self.epochs-self.max_epoch)
            factor = 1. - factor

        self.current_val = factor*(self.max_val-self.min_val)+self.min_val

    def set(self,loss_dict:LossDict):
        loss_dict.weights["aux_sparse"] = self.current_val
        return loss_dict


    

def train(model, optimizer, loss_fun, dataloader, device,
        loss_scheduler = None,
          gradient_max=None,mixed_precision=False):
    try:
        tracker = StatsTracker()
        model.train()
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            batch:BatchData = batch.to(device,non_blocking=True)
            pattern = batch.pattern 
            freqs = batch.freqs
            phy_para = batch.phy_para
            tgt_z_vel = batch.z_vel


            with record_function("model_forward"):

                with torch.amp.autocast(device_type="cuda",enabled=mixed_precision,dtype=torch.bfloat16):
                    pred_z_vel,modedynamics = model(pattern,phy_para.to_dict(),freqs)
                    loss = loss_fun(pred_z_vel, tgt_z_vel,modedynamics)
                    if loss_scheduler is not None:
                        loss = loss_scheduler.set(loss)
                    loss_value = loss.sum()


            with record_function("model_backward"):
                loss_value.backward()



                if(gradient_max is not None):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_max,error_if_nonfinite=True)
                    #torch.nn.utils.clip_grad_value_(model.parameters(),gradient_max)
                else: 
                    grad_norm = None

                optimizer.step()
            
            with record_function("train log"):
                tracker.update(loss.get_values(),pattern.shape[0])
                tracker.update({"loss":loss_value.item()},pattern.shape[0])
                if grad_norm is not None:
                    tracker.update({"grad":grad_norm.item()},pattern.shape[0])
                with torch.no_grad():
                    tgt_frf = field_dict2frf(tgt_z_vel,normalize=True)
                    pred_frf = field_dict2frf(pred_z_vel,normalize=True)
                    metrics = calc_metrics(pred_frf,tgt_frf)
                with torch.no_grad(): 
                    pred_ln_mag = pred_z_vel["ln_mag"]
                    #pred_ln_mag shape: B x F x W x H
                    pred_ln_mag = torch.flatten(pred_ln_mag,start_dim=2)
                    pred_ln_mag = pred_ln_mag - pred_ln_mag.mean(dim=2,keepdim=True)

                    pred_std = pred_ln_mag.std(dim=1).mean()
                    metrics["field_std"] = pred_std
                    tracker.update(metrics,pattern.shape[0])
    
        train_stats= tracker.finish()
        if loss_scheduler is not None:
            train_stats["sparse_coef"] = loss_scheduler.current_val
        return train_stats

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        optimizer.zero_grad(set_to_none=True)
        raise KeyboardInterrupt
    




def run_training(model, trainloader,valloader,
                 setup_optimizer=torch.optim.Adam,
                 setup_scheduler = None,
                 loss_fun=torch.nn.MSELoss(),
                 device=torch.device("cpu"),
                 num_epoch=100,
                 gradient_max=None,
                 logger : Logger  = None,
                 mixed_precision = False,
                 loss_scheduler = None,
                 ):
    model = model.to(device,non_blocking=True)

    loss_fun = loss_fun
    optimizer = setup_optimizer(model)
    if(setup_scheduler is not None):
        scheduler = setup_scheduler(optimizer)
    else:
        scheduler = None

    if(device.type == 'cuda'):
        torch.backends.cudnn.benchmark = True

     
    try:
        for epoch in range(num_epoch):
            with record_function("train"):
                train_stats = train(model,optimizer,loss_fun,trainloader,device,
                    gradient_max=gradient_max,mixed_precision=mixed_precision,
                   loss_scheduler=loss_scheduler, 
                    )
                

            with record_function("evaluate"):
                eval_stats = evaluate_model(model,valloader,device=device)

            with record_function("epoch_log") :
                if logger is not None:
                    logger.log(epoch,train_stats , eval_stats,model,
                            {
                                f"lr": optimizer.param_groups[0]["lr"]
                            }
                            )            
            
            if scheduler is not None:
                scheduler.step(epoch)
            
            if loss_scheduler is not None:
                loss_scheduler.step(epoch)
            

    except KeyboardInterrupt:
        pass

    return model
