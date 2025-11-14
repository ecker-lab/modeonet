import numpy as np
from vibromodes.globals import AMPLITUDE_STD
from vibromodes.kirchhoff import DEFAULT_PLATE_PARAMETERS, PlateParameter
from vibromodes.models.ModeONet import (
    LinearSuperposition, ModeONet,complex2log_mag_dir,
    LogSuperposition,
)
import torch
import pytest
from vibromodes.velocity_field import field_dict2linear





def test_spatial_temporal_shapes():
    B = 4
    F = 6

    output_size = (61,91)
    pattern = torch.rand((B,61,91))

    phy_para = PlateParameter.from_array(
        torch.tensor(DEFAULT_PLATE_PARAMETERS).unsqueeze(0).repeat(B,1)
    )

    frequencies = torch.rand((B,F))

    model = ModeONet(8,12,output_size=output_size)

    tgt_field,_ = model(pattern,phy_para.to_dict(),frequencies)

    tgt_field = tgt_field["ln_mag"]
    assert tgt_field.shape == (B,F,output_size[0],output_size[1])

def test_log_superposition():
    
    B,F,n_peaks = 9,7,5
    W,H = 3,2
    mode_responses = torch.rand((B,F,n_peaks),dtype=torch.complex64)
    mode_shapes = torch.rand((B,n_peaks,W,H),dtype=torch.complex64)
    mode_shapes_normed =mode_shapes / torch.linalg.norm(mode_shapes.view(B,n_peaks,W*H),dim=-1).unsqueeze(-1).unsqueeze(-1)

    frequencies = torch.rand((B,F))


    expected_field = (mode_shapes_normed[:,None,:,:,:]*
                mode_responses[:,:,:,None,None]).sum(dim=2).to(torch.complex64)
    
    
    

    log_superposition = LogSuperposition(correct_distribution=False) 
    bias = log_superposition.bias[frequencies]
    factor = (-bias*0.5).exp()*n_peaks

    normed_respones  =log_superposition.mode_response_physical2normed(mode_responses,frequencies)
    normed_shapes  =log_superposition.mode_shape_physical2normed(mode_shapes)


    expected_sparse_loss = torch.log(torch.linalg.norm(mode_responses*factor[:,:,None],dim=1)+1).mean()

    pred_field,aux_loss = log_superposition.forward(normed_shapes,normed_respones,frequencies)

    actual_field = field_dict2linear(pred_field)


    torch.testing.assert_close(actual_field,expected_field)
    torch.testing.assert_close(
        aux_loss["aux_sparse"],expected_sparse_loss
    )
    expected_ln_mag = expected_field.abs().log()*2/AMPLITUDE_STD
    expected_normed_field = expected_field/torch.linalg.norm(expected_field.flatten(start_dim=-2),dim=-1)[:,:,None,None]

    torch.testing.assert_close(
        pred_field["ln_mag"],expected_ln_mag
    )

    torch.testing.assert_close(
        pred_field["normed_field"][:,:,0],torch.real(expected_normed_field)
    )

    torch.testing.assert_close(
        pred_field["normed_field"][:,:,1],torch.imag(expected_normed_field)
    )


def test_mode_normalization():
    superposition = LogSuperposition(correct_distribution=False)
    
    B,F,n_peaks = 7,5,4
    W,H = 61,91
    spatial_part = torch.rand((B,n_peaks,W,H),dtype=torch.complex64)

    norm = torch.linalg.norm(spatial_part.view(B,n_peaks,W*H),dim=-1)
    
    excepted_complex = spatial_part/norm.unsqueeze(-1).unsqueeze(-1)
    expected_ln_mag = excepted_complex.abs().log()

    

    spatial_part_ln = spatial_part.abs().log()
    spatial_part_dir = spatial_part/torch.abs(spatial_part)
    spatial_part_dir_real = torch.real(spatial_part_dir)
    spatial_part_dir_imag = torch.imag(spatial_part_dir)

    actual = superposition._normalize_mode(spatial_part_ln)

    torch.testing.assert_close(
        actual,expected_ln_mag
    )


def test_linear_superposition():
    B,F,n_peaks = 15,30,20
    W,H = 50,40
    mode_responses = torch.ones((B,F,n_peaks),dtype=torch.complex64)
    mode_shapes = torch.ones((B,n_peaks,W,H),dtype=torch.complex64)
    mode_shapes_normed =mode_shapes / torch.linalg.norm(mode_shapes.view(B,n_peaks,W*H),dim=-1).unsqueeze(-1).unsqueeze(-1)

    frequencies = torch.ones((B,F))


    expected_field = (mode_shapes_normed[:,None,:,:,:]*
                mode_responses[:,:,:,None,None]).sum(dim=2).to(torch.complex64)
    
    
    

    superposition = LinearSuperposition() 
    bias = superposition.bias[frequencies]
    factor = (-bias*0.5).exp()*n_peaks

    normed_respones  =superposition.mode_response_physical2normed(mode_responses,frequencies)
    normed_shapes  =superposition.mode_shape_physical2normed(mode_shapes)


    expected_sparse_loss = torch.log(torch.linalg.norm(mode_responses*factor[:,:,None],dim=1)+1).mean()

    pred_field,aux_loss = superposition.forward(normed_shapes,normed_respones,frequencies)

    expected_ln_mag = expected_field.abs().log()*2/AMPLITUDE_STD
    expected_normed_field = expected_field/torch.linalg.norm(expected_field.flatten(start_dim=-2),dim=-1)[:,:,None,None]

    torch.testing.assert_close(
        pred_field["ln_mag"],expected_ln_mag
    )

    torch.testing.assert_close(
        pred_field["normed_field"][:,:,0],torch.real(expected_normed_field)
    )

    torch.testing.assert_close(
        pred_field["normed_field"][:,:,1],torch.imag(expected_normed_field)
    )

    actual_field = field_dict2linear(pred_field)


    torch.testing.assert_close(actual_field,expected_field)
    torch.testing.assert_close(
        aux_loss["aux_sparse"],expected_sparse_loss
    )
    pass

def test_log_superposition_mode_normalization_mean():
    
    superposition = LogSuperposition(correct_distribution=True)
    B,F,n_peaks = 7,5,4
    W,H = 61,91
    spatial_part_ln = torch.randn((B,n_peaks,W,H),dtype=torch.float32)


    actual = superposition._normalize_mode(spatial_part_ln)
    
    assert actual.mean()  == pytest.approx(0,abs=1.5)
    assert actual.std()  == pytest.approx(1,abs=.5)



def test_linear_superposition_mode_normalization_mean():
    
    superposition = LinearSuperposition(correct_distribution=True)
    B,F,n_peaks = 7,5,4
    W,H = 61,91
    mode_shapes = torch.randn((B,2,n_peaks,W,H),dtype=torch.float32)


    actual = superposition._normalize_mode(mode_shapes)
    
    assert actual.mean()  == pytest.approx(0,abs=1.5)
    assert actual.std()  == pytest.approx(1,abs=.5)


def test_log_superposition_momemts():
    superposition = LogSuperposition(correct_distribution=True)

    B,F,n_peaks = 16,20,16
    W,H = 61,91

    mode_shapes = torch.randn(B,3,n_peaks,W,H)
    mode_responses = torch.randn(B,F,3,n_peaks)

    frequencies = torch.randn(B,F)

    field,aux_loss = superposition(mode_shapes,mode_responses,frequencies)

    ln_mag = field["ln_mag"] - superposition.bias[frequencies][:,:,None,None]/AMPLITUDE_STD


    assert ln_mag.mean()  == pytest.approx(0,abs=1.5)
    assert ln_mag.std()  == pytest.approx(0.5,abs=.4)


def test_linear_superposition_momemts():
    superposition = LinearSuperposition(correct_distribution=True)

    B,F,n_peaks = 16,20,16
    W,H = 61,91

    mode_shapes = torch.randn(B,2,n_peaks,W,H)
    mode_responses = torch.randn(B,F,2,n_peaks)

    frequencies = torch.randn(B,F)

    field,aux_loss = superposition(mode_shapes,mode_responses,frequencies)

    ln_mag = field["ln_mag"] - superposition.bias[frequencies][:,:,None,None]/AMPLITUDE_STD


    assert ln_mag.mean()  == pytest.approx(0,abs=1.5)
    assert ln_mag.std()  == pytest.approx(0.5,abs=.4)
