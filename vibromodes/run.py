import argparse
import random
import numpy as np
import pandas as pd
import subprocess
import torch
from torchinfo import summary
import yacs
from vibromodes.hdf5_dataset import AugmentationTransform, BatchData, Hdf5Dataset,create_train_eval_dataloaders
from vibromodes.config import default_config
from vibromodes.loss import load_loss
from vibromodes.metrics import evaluate_model
from vibromodes.models import FQOUNet, load_model
from vibromodes.train import (
    run_training, SparseScheduler
    )
from vibromodes.log import Logger
from torch.optim import Adam,AdamW

from timm.scheduler import CosineLRScheduler

from torch.profiler import profile, ProfilerActivity, record_function

import logging
torch._logging.set_logs(graph_breaks=True,dynamo=50)
import os
import shutil

def load_data(config,seed = 42):
    data_config = config.data
    transforms = AugmentationTransform(data_config.random_flip,data_config.noise)
    data_paths = []

    for path in data_config.train_paths:
        if "LOCAL_TMPDIR" in os.environ:
            file_name = os.path.basename(path)
            tmp_dir = os.environ["LOCAL_TMPDIR"]
            data_path = os.path.join(tmp_dir,file_name)
            if not os.path.isfile(data_path):
                print(f"copy data to {data_path}")
                shutil.copy(path,tmp_dir)
            else:
                print("data already in tmp directory")
        else:
            print("no tmp dir found")
            data_path = path
        data_paths.append(data_path)

    train_dataset = Hdf5Dataset(data_paths,precision=config.precision,vel_field_size=config.mode_size,
                                transforms=transforms,dataset_limit=data_config.dataset_limit,
                                in_memory=data_config.in_memory)
    return create_train_eval_dataloaders(
        train_dataset, 
        train_eval_split=data_config.train_eval_split,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        seed=seed,
        pin_memory=True,
        drop_last=data_config.train_drop_last,
        random_split=data_config.random_split,
    )
def load_configs(paths=[]):
    config = default_config()
    for path in paths:
        config.merge_from_file(path)
    return config 



def setup_optimizer(config,model):
    if config.type=="Adam":
        return Adam(model.parameters(),lr=config.lr)
    if config.type=="AdamW":
        return AdamW(model.parameters(),lr=config.lr,
                               weight_decay=config.weight_decay,fused=True)
    else:
        raise NotImplementedError(f"Optimizer {config.type} not implemented")

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.benchmark = False


def get_model_summary(model,train_loader,device):
    for batch in train_loader:
        break
    print("test",len(train_loader))

    batch :BatchData = batch.to(device,non_blocking=True)
    return summary(model,
                    input_data=[batch.pattern,batch.phy_para.to_dict(),batch.freqs],
                )


def setup_scheduler(config,optimizer):
    if(config.scheduler.type=="none"):
        return {}
    if(config.scheduler.type=="CosLR"):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.cycle_length,
            eta_min=config.scheduler.min_lr,
        )
        
    if(config.scheduler.type=="CosLRWarmup"):
        return CosineLRScheduler(
            optimizer,t_initial=config.scheduler.cycle_length,
            lr_min=config.scheduler.min_lr,
            cycle_decay=0.1,
            warmup_lr_init=config.scheduler.warmup_lr,
            warmup_t=config.scheduler.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True)

            

def setup_loss_scheduler(config):
    if config.loss_scheduler.type == "none":
        return None
    elif config.loss_scheduler.type == "Sparse":
        return SparseScheduler(
            epochs=config.epochs,
            max_epoch=config.loss_scheduler.max_epoch,
            min_val=config.loss_scheduler.min_val, 
            max_val=config.loss_scheduler.max_val,
        )
    else:
        raise ValueError(f"{config.loss_scheduler.type} is not supported as a Loss Scheduler type.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs",type=str,nargs="*")
    parser.add_argument("-d","--device",choices=["cuda","cpu"],default="cuda")
    parser.add_argument("-w","--wandb",action="store_true")
    parser.add_argument("-e","--epochs",type=int,default=None)
    parser.add_argument("-lr","--learning_rate",type=float,default=None)
    parser.add_argument("-clr","--cos_lr",type=float,default=None)
    parser.add_argument("-t","--type",type=str,default=None)
    parser.add_argument("-n","--run_name",type=str,default=None)
    parser.add_argument("--dataset_limit",type=int,default=None)
    parser.add_argument("--batch_size",type=int,default=None)
    parser.add_argument("--train_eval_split",type=float,default=None)
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("-i","--in_memory",action="store_true")
    parser.add_argument("--no_augmentation",action="store_true")
    parser.add_argument("--weight_decay",type=float,default=None)
    parser.add_argument("--optimizer",type=str,default=None)
    parser.add_argument("--temporal_dim",type=int,default=None)
    parser.add_argument("--phase",type=float,default=None)
    parser.add_argument("--sparse",type=float,default=None)
    parser.add_argument("--linear_rep",action="store_true")
    parser.add_argument("--cycle_length",type=int,default = None)
    parser.add_argument("--warmup",type=int,default=None)
    parser.add_argument("--query_conditioning",choices=["bias","bias+scale","film"],default=None)
    parser.add_argument("--analytic_response",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--grad_max",type=float,default=None)

    
    args = parser.parse_args()


    config = load_configs(args.configs)

    if args.epochs is not None:
        config.epochs = args.epochs
    
    if args.learning_rate is not None:
        config.optimizer.lr = args.learning_rate
    
    if args.type is not None:
        config.log.job_type = args.type
    
    if args.run_name is not None:
        config.log.run_name = args.run_name
    
    if args.dataset_limit is not None:
        config.data.dataset_limit = args.dataset_limit

    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    
    if args.train_eval_split is not None:
        config.data.train_eval_split = args.train_eval_split
    
    if args.in_memory:
        config.data.in_memory = args.in_memory
    
    if args.cos_lr is not None:
        config.scheduler.min_lr = args.cos_lr
    
    if args.weight_decay is not None:
        config.optimizer.weight_decay = args.weight_decay
    
    if args.optimizer is not None:
        config.optimizer.type = args.optimizer
    
    if args.no_augmentation:
        config.data.random_flip = False
        config.data.noise = 0.
    
    if args.temporal_dim:
        config.model.temporal_dim = args.temporal_dim
    
    if args.phase is not None:
        config.loss.weights.phase = args.phase
    
    if args.sparse is not None:
        config.loss.weights.aux_sparse = args.sparse
    
    if args.linear_rep:
        config.model.linear_rep = args.linear_rep
    
    if args.cycle_length is not None:
        config.scheduler.cycle_length = args.cycle_length
    if args.warmup is not None:
        config.scheduler.warmup_epochs = args.warmup

    if args.analytic_response is not None:
        config.model.analytic_response = args.analytic_response
    
    if args.query_conditioning is not None:
        config.model.query_conditioning = args.query_conditioning
    
    if args.grad_max is not None:
        config.gradient_max = args.grad_max

    config.log.wandb = args.wandb

    set_global_seed(args.seed)
    if args.device=="cpu":
        device = torch.device("cpu")
    elif args.device=="cuda":
        device = torch.device("cuda")
        #torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    logger = Logger(config,device)


    train_loader, eval_loader = load_data(config)

    model = load_model(config)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    loss_fn = load_loss(config).to(device)


    loss_scheduler = setup_loss_scheduler(config)


    

    model = model.to(device=device)

    model_summary =  get_model_summary(model,train_loader,device)



    #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True,profile_memory=True) as prof:
    print("Training started") 
    run_training(model,train_loader,eval_loader,
                setup_optimizer=
                    lambda p: setup_optimizer(config.optimizer,p),
                setup_scheduler= lambda o : setup_scheduler(config,o),
                device=device,
                loss_fun=loss_fn,
                num_epoch=config.epochs,
                logger = logger,
                gradient_max=config.gradient_max,
                mixed_precision=config.mixed_precision,
                loss_scheduler=loss_scheduler,
                ) 
    
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # prof.export_chrome_trace("./tmp/trace.json")
    #prof.export_memory_timeline("./tmp/memory_figure.html",device="cuda:0")

    logger.save_model(model,"last_model")

    print("Training finished") 
    train_metrics = evaluate_model(model,train_loader,device)
    eval_metrics = evaluate_model(model,eval_loader,device)
    
    logger.log_final_metrics("last_model",train_metrics,eval_metrics)


    model = logger.load_model(model,"best_mse_model")
    

    train_metrics = evaluate_model(model,train_loader,device)
    eval_metrics = evaluate_model(model,eval_loader,device)
    logger.log_final_metrics("best_mse_model",train_metrics,eval_metrics)

    logger.finish()