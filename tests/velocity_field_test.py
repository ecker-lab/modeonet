from vibromodes.velocity_field import (
    field_dict2frf,field_dict2linear,
    field_linear2dict,field_linear2frf,
    field_ln_mag_dir2dict,
    field_linear2ln_mag_dir,
)
import torch

#TODO test this
def test_field_conversion():
    B,F,W,H = 7,5,3,2
    field = torch.randn(B,F,W,H) + 1.j* torch.randn(B,F,W,H)

    field_dict = field_linear2dict(field)
    field2 = field_dict2linear(field_dict)

    torch.testing.assert_close(field,field2)

    normed = field_dict["normed_field"]
    norm  = torch.linalg.norm(normed.view(B,F,-1),dim=-1)

    torch.testing.assert_close(norm,torch.ones_like(norm))

def test_frfs():
    B,F,W,H = 7,5,3,2
    field = torch.randn(B,F,W,H) + 1.j* torch.randn(B,F,W,H)

    field_dict = field_linear2dict(field)

    frf1 = field_linear2frf(field)
    frf2 = field_dict2frf(field_dict)

    torch.testing.assert_close(frf1,frf2)

    

def test_ln_mag_dir():
    B,F,W,H = 7,5,3,2
    field = torch.randn(B,F,W,H) + 1.j* torch.randn(B,F,W,H)

    field_ln_mag_dir = field_linear2ln_mag_dir(field)

    field_dict = field_ln_mag_dir2dict(field_ln_mag_dir)
    field2 = field_dict2linear(field_dict)

    torch.testing.assert_close(field,field2)