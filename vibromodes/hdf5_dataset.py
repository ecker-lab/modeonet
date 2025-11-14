import torch
from torch import nn
import hdf5plugin
import h5py
from torch.utils.data import Dataset, DataLoader, get_worker_info, random_split
import numpy as np
from torchvision.transforms import Resize

from vibromodes.kirchhoff import DEFAULT_PLATE_PARAMETERS, PlateParameter

from tensordict import tensorclass,TensorDict

from torch.profiler import profile, ProfilerActivity, record_function
from vibromodes.velocity_field import field_linear2dict

DO_ASSERTS=False


@tensorclass
class BatchData:
    #shape ... x  W x H
    #this is normalized
    pattern : torch.Tensor

    
    #shape ... x F
    #this is normalized
    freqs: torch.Tensor

    phy_para : PlateParameter

    #shape ... x F x W x H
    z_vel : TensorDict

    #shape ...
    id : torch.Tensor

    def set_precision(self,precision):
        if precision == 32:
            self.to(device="cpu",dtype=torch.float32)
        else:
            self.to(device="cpu",dtype=torch.float64)
        return self

    def to(self,device=None,non_blocking=False,dtype=None):
        self.pattern = self.pattern.to(device=device,dtype=dtype,non_blocking=non_blocking)
        self.freqs = self.freqs.to(device=device,dtype=dtype,non_blocking=non_blocking)
        for key in self.z_vel.keys():
            self.z_vel[key] = self.z_vel[key].to(device=device,dtype=dtype,non_blocking=non_blocking)
        self.phy_para = self.phy_para.to(device,non_blocking=non_blocking,dtype=dtype)
        #we won't cast the id
        self.id = self.id.to(device=device,non_blocking=non_blocking)
        return self



class AugmentationTransform(nn.Module):
    def __init__(self,random_flip=False,noise=0.):
        super().__init__()
        self.random_flip = random_flip
        self.noise=noise

    def flip_augmentation(self,sample:BatchData):
        assert len(sample.pattern.shape)==3
        if not self.random_flip:
            return sample

        flip_x = torch.rand(sample.batch_size)<0.5
        flip_y = torch.rand(sample.batch_size)<0.5

        return self.flip_on_mask(sample,flip_x,flip_y)



    def flip_on_mask(self,sample:BatchData,flip_x:torch.Tensor,flip_y:torch.Tensor):
        #flip_x
        sample.pattern[flip_x] = sample.pattern[flip_x].flip(-2)

        sample.phy_para.force_x[flip_x] = -sample.phy_para.force_x[flip_x]

        for key in sample.z_vel.keys():
            sample.z_vel[key][flip_x] = sample.z_vel[key][flip_x].flip(-2)

        #flip_y
        sample.pattern[flip_y] = sample.pattern[flip_y].flip(-1)

        sample.phy_para.force_y[flip_y] = -sample.phy_para.force_y[flip_y]

        for key in sample.z_vel.keys():
            sample.z_vel[key][flip_y] = sample.z_vel[key][flip_y].flip(-1)


        return sample
    
    def noise_augmentation(self,sample):
        pattern = sample.pattern
        if self.noise==0:
            return sample
        
        noise_levels = torch.rand(sample.batch_size) * self.noise 
        noise_levels = noise_levels.unsqueeze(1).unsqueeze(1)

        gaussian = torch.randn_like(pattern)

        noisy_image = (1 - noise_levels) * pattern + gaussian * noise_levels # this matches the OT noise formulation
        sample.pattern = noisy_image
        return sample







    def forward(self,sample):
        sample = self.flip_augmentation(sample)
        sample = self.noise_augmentation(sample)
        return sample

def get_local_indices(ids,end_ids):
    local_ids = []
    last_end_id = 0
    for end_id in end_ids:
        i = np.searchsorted(ids,end_id,side="left")
        local_ids.append(ids[:i]-last_end_id)
        ids = ids[i:]
        last_end_id = end_id
    return local_ids

class FileDispatcher:
    def __init__(self,paths,dataset_limit,preprocess=None):
        self.total_length = 0
        self.file_lengths = []
        
        self.keys = set(["frequencies","phy_para","bead_patterns","z_vel"])

        for path in paths:
            with h5py.File(path,"r") as file:
                keys = set(file.keys())
                assert self.keys.issubset(keys)

                file_length = file["phy_para"].shape[0]
                if self.total_length+file_length > dataset_limit:
                    file_length = dataset_limit-self.total_length
                    assert file_length > 0
                
                self.file_lengths.append(file_length)
                self.total_length += file_length

            if self.total_length >= dataset_limit:
                break


        self.paths = paths
        self.file_end_indices = np.cumsum(self.file_lengths)
        self.preprocess = preprocess

    def __len__(self):
        return self.total_length

    def preload_all_files(self):
        self.files = []
        for file_length,path in zip(self.file_lengths,self.paths):
            with h5py.File(path,"r") as file:
                tmp_file = {key: file[key][:file_length] for key in self.keys}
                
            if self.preprocess is not None:
                tmp_file = self.preprocess(tmp_file)

            self.files.append(tmp_file)
        
        print("Resizing dataset done")
        self.preprocess = None
        #self.file_lengths = [self.total_length]
        #self.file_end_indices = np.array([self.total_length])
    
    def open_files(self):
        if hasattr(self,"files"):
            #print("file is already opened")
            return
        
        worker_info = get_worker_info()

        if worker_info is None:
            print("init file from main thread")
        else:
            print(f"init file from worker {worker_info.id}")

        self.files = []
        for path in self.paths: 
            self.files.append(h5py.File(path,"r"))
    
    def get(self,idx):
        #idx needs to be sorted
        self.open_files()


        local_ids = get_local_indices(idx,self.file_end_indices)

        batch = {}
        
        for key in self.keys:
            data = [
                file[key][local_id] for file,local_id in zip(self.files,local_ids)
            ]
            batch[key] = np.concatenate(data)
        
        if self.preprocess is not None:
            batch = self.preprocess(batch)
        return batch


                

class Hdf5Dataset:
    def __init__(self,paths,precision=32,vel_field_size=None,
                 transforms=nn.Identity(),dataset_limit=None,in_memory=False):


        self.precision = precision
        self.paths = paths
        self.transforms = transforms
        self.vel_field_size = vel_field_size

        if(vel_field_size is not None):
            self.resizer = Resize(vel_field_size)
        else:
            self.resizer = None


        def preprocess(batch):
            batch["z_vel"] = self.resize_field(batch["z_vel"])
            return batch
        
        self.files = FileDispatcher(paths,dataset_limit,preprocess)

        if in_memory:
            self.files.preload_all_files()



    def resize_field(self,field):
        
        if self.resizer is None:
            return field
        
        if self.vel_field_size[0]==field.shape[0] and self.vel_field_size[1] == field.shape[1]:
            return field
        
        field = torch.from_numpy(field)
        real_field = self.resizer(torch.real(field))
        imag_field = self.resizer(torch.imag(field))

        result = real_field + imag_field*1.j
        return result.numpy()




    def __getitems__(self,idx):


        #we have to sort the ids,
        # since hdf5 wants an increasing order of indices
        idx = np.array(idx)
        idx.sort()
            
        batch = self.files.get(idx)

        with record_function("load_data"):
            for attempts in range(5):
                try:
                    freqs = torch.from_numpy(batch["frequencies"])
                    data = BatchData(
                        pattern=torch.from_numpy(batch["bead_patterns"]),
                        freqs=freqs,
                        phy_para=PlateParameter.from_array(
                            torch.from_numpy(batch["phy_para"])
                            ),

                        z_vel = TensorDict(
                            field_linear2dict(
                                torch.from_numpy(batch["z_vel"])
                            ),
                            batch_size = [len(idx)]
                        ),
                        id = torch.tensor(idx),
                        batch_size = [len(idx)] ,
                    )
                except Exception as e:
                    print("Error during loading data") 
                    print("Indicies:")
                    print(idx)
                    print("Error message")
                    print(str(e))
                    #try again
                else:
                    #we successfully loaded the data
                    break
            else:
                raise Exception("we cound't load the data")



                
                
        with record_function("data augmentation"):
            data.set_precision(self.precision)
            data.pattern = (data.pattern/0.02)*2.-1
            data.freqs = normalize_frequencies_(data.freqs)
            data.phy_para = normalize_boundary_condition_(data.phy_para)
            data.phy_para = normalize_force_(data.phy_para)

            data = self.transforms(data)

        return data
    
    def __getitem__(self,idx):

        is_scaler = type(idx) is int \
                or (torch.is_tensor(idx) and len(idx.shape)==0)\
                or np.isscalar(idx)
        if is_scaler:
            idx = np.array([idx])
            batch = self.__getitems__(idx)
            return batch[0]
        else:
            return self.__getitems__(idx)


    def __len__(self):
        return len(self.files)




def create_train_eval_dataloaders(
    dataset,
    train_eval_split,
    batch_size=1,
    num_workers=0,
    collate_fn=lambda x: x,
    seed=42,
    dataset_limit=None,
    pin_memory=False,
    drop_last=True,
    random_split = True,
):
    assert train_eval_split >= 0.0
    assert train_eval_split <= 1.0

    if dataset_limit is not None and dataset_limit<len(dataset):
        indices = np.arange(0,dataset_limit)

        dataset = torch.utils.data.Subset(dataset,indices)

    N = len(dataset)
    train_length = int(train_eval_split * N)
    eval_length = N - train_length

    generator = torch.Generator().manual_seed(seed)

    if random_split:
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_length, eval_length], generator=generator
        )
    else:
        train_dataset = torch.utils.data.Subset(dataset,[0,1])
        eval_dataset = torch.utils.data.Subset(dataset,np.arange(train_length,N))


    print(f"Trainset size: {len(train_dataset)} | Evalset size: {len(eval_dataset)} ")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=True,
        persistent_workers=num_workers>0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers>0,
    )
    return train_loader, eval_loader

def normalize_boundary_condition_(phy_para:PlateParameter):
    phy_para.boundary_condition-=50.
    phy_para.boundary_condition/=28.8675
    return phy_para

def normalize_force_(phy_para):
    phy_para.force_x-=0.5
    phy_para.force_x/=0.173205
    phy_para.force_y-=0.5
    phy_para.force_y/=0.173205

    return phy_para

def normalize_frequencies_(frequencies):
    frequencies/=300.
    frequencies*=2.
    frequencies-=1.
    return frequencies

def normalize_frequencies(frequencies):
    return frequencies/300. *2. -1.

def load_testset(test_set_path):


    with h5py.File(test_set_path) as data:

        B = data["frequencies"].shape[0]
        phy_para = torch.tensor(DEFAULT_PLATE_PARAMETERS).unsqueeze(0).repeat((B,1))


        phy_para = PlateParameter.from_array(phy_para)
        phy_para.force_x = torch.from_numpy(data["phy_para"][:,2])
        phy_para.force_y = torch.from_numpy(data["phy_para"][:,1])
        phy_para.boundary_condition = torch.from_numpy(data["phy_para"][:,0])

        freqs = torch.from_numpy(data["frequencies"][:])

        pattern = torch.from_numpy(data["bead_patterns"][:])
        pattern = (pattern/0.02)*2.-1.

        dataset = BatchData(
            pattern,
            freqs,
            phy_para,
            TensorDict(
                field_linear2dict(torch.from_numpy(data["z_vel_abs"][:])),
                    batch_size=[B]),
            torch.arange(B),
            batch_size= [B]
        )
    dataset.set_precision(32)

    dataset.freqs = normalize_frequencies_(dataset.freqs)
    dataset.phy_para = normalize_boundary_condition_(dataset.phy_para)
    dataset.phy_para = normalize_force_(dataset.phy_para)
    return dataset