
from vibromodes.kirchhoff import PlateParameterIndex,PlateParameter
from vibromodes.modes import (
    calc_mode_response,calc_mode_response_log_mag,
    calc_mode_response_complex
    

)
import torch


def test_mode_response_log():
    loss_factor = torch.tensor([0.1])

    eigenfreq = torch.tensor([100]).float()

    frequencies = torch.arange(1,300).float()

    #component = 1.j*frequencies/(eigenfreq**2 * (1+1.j*loss_factor) - frequencies**2)
    component = calc_mode_response_complex(frequencies,eigenfreq**2,loss_factor)
    expected_component_ln_mag = torch.log(torch.absolute(component))
    expected_component_phase = torch.angle(component)

    actual_ln_mag,actual_phase = calc_mode_response_log_mag(frequencies,eigenfreq**2,loss_factor)

    torch.testing.assert_close(actual_ln_mag,expected_component_ln_mag)
    torch.testing.assert_close(actual_phase,expected_component_phase)


def test_calc_mode_response():
    loss_factor = torch.tensor([0.1])

    eigenfreq = torch.tensor([100]).float()

    frequencies = torch.arange(1,300).float()

    #component = 1.j*frequencies/(eigenfreq**2 * (1+1.j*loss_factor) - frequencies**2)
    component = calc_mode_response_complex(frequencies,eigenfreq**2,loss_factor)

    actual_real,actual_imag = calc_mode_response(frequencies,eigenfreq**2,loss_factor)

    torch.testing.assert_close(actual_real,torch.real(component))
    torch.testing.assert_close(actual_imag,torch.imag(component))
    pass