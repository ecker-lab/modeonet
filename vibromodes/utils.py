import torch
from torch import nn

from torchinterp1d import interp1d

def element_size(tensor_class):
    element_size = 0
    for key in tensor_class.keys():
        element_size += tensor_class.get(key).data.element_size()
    return element_size

class convert_1d_to_interpolator(nn.Module):
    def __init__(self, array, min_val, max_val):
        super().__init__()

        self.register_buffer("array",array)
        self.min_val = min_val
        self.max_val = max_val
        x = torch.linspace(min_val, max_val, steps=array.shape[0], device=array.device)
        self.register_buffer("x",x)

    def __getitem__(self, xnew):
        if not isinstance(xnew, torch.Tensor):
            xnew = torch.tensor(xnew, dtype=torch.float32, device=self.array.device)
        original_shape = xnew.shape
        xnew_flat = xnew.flatten()
        interpolated_values_flat = interp1d(self.x, self.array, xnew_flat, None)
        return interpolated_values_flat.view(original_shape)
