import subprocess
from time import time
from typing import Optional
from matplotlib import animation, pyplot as plt
import numpy as np
import torch
import wandb
import os
from pathlib import Path
from yacs.config import CfgNode as CN
import pandas as pd
from vibromodes.hdf5_dataset import BatchData
from vibromodes.kirchhoff import tr_velocity_field_to_frequency_response
from scipy.signal import find_peaks

def default_logger_config():
    config = CN()
    config.job_type = "debug"
    config.run_name = "modeonet"
    config.wandb = False
    config.save_interval = 10
    config.save_path = "Please set the save path in the config!"
    return config


def print_git_info():
    try:
        commit_info = subprocess.check_output(['git', 'log', '-1']).decode('utf-8').strip()

        # Check for modified files (not staged or staged)
        status_info = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip()
        
        print(commit_info)
        print(status_info)


    except subprocess.CalledProcessError as e:
        print("Failed to get Git information")


class Logger:
    def __init__(self,config,device):
        wandb.login()

        run = wandb.init(
            project="vibromodes",
            config=config,
            job_type=config.log.job_type,
            mode=None if config.log.wandb else "disabled",
            name=config.log.run_name,
            settings=wandb.Settings(_disable_stats=True),
        )

        save_dir = os.path.join(config.log.save_path,config.log.job_type)
        Path(save_dir).mkdir(parents=True,exist_ok=True)

        if config.log.job_type != "debug":
            save_dir = os.path.join(save_dir, wandb.run.name)

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # save config to file
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            f.write(config.dump())

        self.save_dir = save_dir

        self.best_mse = np.inf
        self.best_mse_epoch = 0
        self.save_interval = config.log.save_interval
        self.device = device
        self.last_time = time()


        

        print_git_info()



    def finish(self):
        wandb.finish()


    @torch.no_grad()
    def log(self,epoch,train_stats,eval_stats,model,debug_stats = {}):

        
        train_stats = {"train_"+key: train_stats[key] for key in train_stats}
        eval_stats = {"eval_"+key: eval_stats[key] for key in eval_stats}

        stats = train_stats | eval_stats

        stats["epoch_time"] = time()-self.last_time

        if epoch==0:
            print("epoch",end="|")
            for key in stats:
                print(key.ljust(12)[:12],end="|")
            print()
        print(f"{epoch:5d}",end="|")
        for key in stats: 
            print(f"{stats[key]:12.4f}",end="|")
        print()

        stats |= debug_stats


        wandb.log(
            stats | {"epoch":epoch}
        )

        if epoch>self.best_mse_epoch+self.save_interval:
            if stats["eval_mse"]<self.best_mse:
                self.best_mse = stats["eval_mse"]
                self.best_mse_epoch = epoch
                self.save_model(model,"best_mse_model")

        self.last_time = time()


    def save_model(self,model,name):
        model.eval()
        torch.save(model.state_dict(), f'{self.save_dir}/{name}.pth')

    def load_model(self,model,name):
        model.eval()
        path = f'{self.save_dir}/{name}.pth'
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

    def log_final_metrics(self,model_name,train_metrics,eval_metrics):
        metrics_table = pd.DataFrame(
            {
                key: [train_metrics[key], eval_metrics[key]]
                for key in train_metrics
            },
            index=["train", "eval"],
        )
        print("Metrics: ",model_name) 
        print(metrics_table)
        metrics_table.to_csv(f"{self.save_dir}/{model_name}_metics.csv")
    

            
