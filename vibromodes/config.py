from yacs.config import CfgNode as CN

from vibromodes.log import default_logger_config
from vibromodes.loss import default_loss_config
from vibromodes.models import default_model_config



def default_data_config():
    data = CN()

    data = CN()
    data.train_paths = ["Please set the path to the traindataset in the config file."]
    data.test_path = ""
    data.batch_size = 16
    data.train_eval_split = 0.9
    data.num_workers = 0
    data.dataset_limit = None
    data.train_drop_last = True
    data.random_split = True
    data.random_flip = False
    data.noise = 0.
    data.in_memory = False


    return data
    

def default_optimizer_config():    
    data = CN()
    data.lr = 1e-3
    data.type="Adam"
    data.weight_decay = 0.01
    return data

def default_scheduler_config():
    config = CN()
    config.type = "none"
    config.cycle_length = 1000
    config.min_lr = 1e-6
    config.warmup_epochs = 50
    config.warmup_lr = 1e-6

    return config

def default_loss_scheduler_config():
    config = CN()
    config.type = "none" #Sparse
    config.min_val = 0.0001
    config.max_val = 0.002
    config.max_epoch = 250
    return config

def default_config():
    config = CN()

    config.precision = 64
    config.epochs = 200
    config.mode_size = (61,91)

    config.data = default_data_config()
    config.model = default_model_config()
    config.optimizer = default_optimizer_config()
    config.loss = default_loss_config()
    config.log = default_logger_config()
    config.scheduler = default_scheduler_config()
    config.loss_scheduler = default_loss_scheduler_config()
    config.gradient_max = None
    config.mixed_precision = False


    return config

