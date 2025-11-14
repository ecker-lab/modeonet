import math
import numpy as np
from dataclasses import dataclass
import torch
from tensordict import tensorclass

from vibromodes.globals import FRF_STD
from typing import Optional

from vibromodes.utils import element_size


@dataclass
class Geometry:
    a: float
    b: float
    h: float
    points_per_wl: int


@dataclass
class MaterialProperties:
    E: float
    rho: float
    eta: float
    alpha: float
    beta: float
    ray: bool
    n_y: float
    rho_L: float
    c_L: float


@dataclass
class Force:
    F: float
    xF: float
    yF: float


def full_factorial_grid(*dimension_samples):

    n_factorial = 1
    for sample in dimension_samples:
        n_factorial = n_factorial * len(sample)

    nodes = np.zeros((n_factorial, len(dimension_samples)))
    meshgrid_mat = np.meshgrid(*dimension_samples)

    for idx, parameter in enumerate(meshgrid_mat):
        nodes[:, idx] = parameter.flatten()

    return nodes


class KirchhoffPlate:

    def __init__(self, geometry, material, force) -> None:
        print("geometry",geometry.__dict__)
        print("material",material.__dict__)
        print("force",force.__dict__)
        self.geo = geometry
        self.mat = material
        self.force = force

        # reference values for logarithmic representation (values of DIN EN 21683)
        self.L_sh_ref = 25e-16

        self.nodes_X = None
        self.nodes_Y = None

    def calc_B_Plate(self):
        """
        Calculates bending stiffness of rectangular plate
        """
        B = (self.mat.E * self.geo.h**3.0) / (12.0 * (1.0 - self.mat.n_y**2.0))
        return B

    def calc_eigenFrq_Plate_forMode_mn(self, mode_m, mode_n):
        """
        Calculates the eigenfrequencies of rectangular plate for requested mode
        """

        a = self.geo.a
        b = self.geo.b
        h = self.geo.h
        rho = self.mat.rho
        B = self.calc_B_Plate()

        if isinstance(mode_m, np.ndarray):
            mode_m = mode_m.reshape(-1, 1)

        if isinstance(mode_n, np.ndarray):
            mode_n = mode_n.reshape(1, -1)

        omega_mn = (
            np.pi**2.0
            * ((mode_m / a) ** 2.0 + (mode_n / b) ** 2.0)
            * np.sqrt(B / (rho * h))
        )
        return omega_mn

    def estimate_reuired_mn(self, freq_max):
        n = np.arange(1, 1000)
        f_n = self.calc_eigenFrq_Plate_forMode_mn(mode_m=1, mode_n=n) / (2 * np.pi)
        f_n_in_range = f_n < freq_max
        n_required = np.ceil(1.5 * np.sum(f_n_in_range))

        m = np.arange(1, 1000)
        f_m = self.calc_eigenFrq_Plate_forMode_mn(mode_m=m, mode_n=1) / (2 * np.pi)
        f_m_in_range = f_m < freq_max
        m_required = np.ceil(1.5 * np.sum(f_m_in_range))

        return m_required, n_required

    def calc_lambda_bending(self, frq_max):
        """
        Calculates the beding wavelength of infinity rectangular plate
        """
        lambda_B = np.sqrt((2.0 * np.pi) / frq_max) * (
            (self.mat.E * self.geo.h**2.0)
            / (12.0 * (1.0 - self.mat.n_y**2.0) * self.mat.rho)
        ) ** (0.25)
        return lambda_B

    def calc_freq_sweep(self, frq, m=None, n=None, verbose=True):
        """
        Calculates the velocity distribution on a force excited rectangular plate (simply-supported) for one frequency
        """
        if isinstance(frq, np.ndarray):
            frq_max = frq[-1]
            n_frq = frq.size
        else:
            frq_max = frq
            n_frq = 1

        # Determine number of required modes in frequency range
        if m is None:
            m_est, n_est = self.estimate_reuired_mn(frq_max)
            m = m_est

        if n is None:
            m_est, n_est = self.estimate_reuired_mn(frq_max)
            n = n_est

        # Generate grid
        lamda_B_min = self.calc_lambda_bending(frq_max)  # min. bending wavelength
        dis_max = (
            lamda_B_min / self.geo.points_per_wl
        )  # max. distance of dicrete points for Rayleigh calc
        elem_x = math.ceil(self.geo.a / dis_max)
        elem_y = math.ceil(self.geo.b / dis_max)
        dist_x = self.geo.a / elem_x
        dist_y = self.geo.b / elem_y
        anz_elem = elem_x * elem_y
        # point coord.
        self.nodes_X = np.linspace(dist_x / 2, self.geo.a - dist_x / 2, elem_x).reshape(
            -1, 1
        )
        self.nodes_Y = np.linspace(dist_y / 2, self.geo.b - dist_y / 2, elem_y).reshape(
            -1, 1
        )

        node_list = full_factorial_grid(self.nodes_X, self.nodes_Y)

        n_nodes = self.nodes_X.shape[0] * self.nodes_Y.shape[0]

        n_array = np.arange(1, n + 1).reshape(1, -1)
        m_array = np.arange(1, m + 1).reshape(-1, 1)
        omega_mn = self.calc_eigenFrq_Plate_forMode_mn(m_array, n_array)
        phi_mn_err = np.sin(m_array * math.pi * self.force.xF / self.geo.a) @ np.sin(
            n_array * math.pi * self.force.yF / self.geo.b
        )

        approx_gbytes = n_array.size * m_array.size * n_nodes * 16 * 10**-9

        if approx_gbytes < 8:
            matrix_comp = True
            # 3D Matrix computation
            phi_xm = np.sin(m_array.T * math.pi * node_list[:, :1] / self.geo.a)
            phi_yn = np.sin(n_array * math.pi * node_list[:, 1:2] / self.geo.b)

            phi_xm = phi_xm[:, :, np.newaxis]
            phi_yn = phi_yn[:, np.newaxis, :]

            phi_mn_xy = np.matmul(phi_xm, phi_yn)
            v1 = phi_mn_xy * phi_mn_err

        else:
            matrix_comp = False

        v = np.zeros((n_frq, n_nodes), dtype=complex)

        # Loop over frequency

        if verbose:
            last_print = 0
            print("Progrress: 0%", end="")

        for idx_f, f in enumerate(frq):

            omega = 2 * math.pi * f

            # if Rayleigh damping is requested
            if self.mat.ray is True:
                Lehr_Damp = (self.mat.alpha / (4.0 * math.pi * f)) + (
                    self.mat.beta * math.pi * f
                )
                self.mat.eta = Lehr_Damp * 2.0

            if matrix_comp:
                v2o = omega_mn**2.0 - omega**2.0 - 1j * self.mat.eta * omega_mn**2.0
                v2u = (
                    omega_mn**2.0 - omega**2.0
                ) ** 2.0 + self.mat.eta**2.0 * omega_mn**4.0
                v2 = v2o / v2u
                # for eta = 0
                # v2 = 1./(omega_mn**2. -omega**2.)
                v_12 = v1 * v2
                sum_v_12 = np.sum(v_12, axis=(1, 2))
                v[idx_f] = (
                    (4.0 * self.force.F * omega)
                    / (self.mat.rho * self.geo.a * self.geo.b * self.geo.h)
                ) * sum_v_12

            else:
                # velocity for every point
                v_xy = np.zeros(n_nodes, dtype=complex)

                for idx, xy_i in enumerate(n_nodes):
                    phi_mn_xy = np.sin(
                        m_array * math.pi * xy_i[0] / self.geo.a
                    ) @ np.sin(n_array * math.pi * xy_i[1] / self.geo.b)
                    v1 = phi_mn_err * phi_mn_xy
                    v2o = omega_mn**2.0 - omega**2.0 - 1j * self.mat.eta * omega_mn**2.0
                    v2u = (
                        omega_mn**2.0 - omega**2.0
                    ) ** 2.0 + self.mat.eta**2.0 * omega_mn**4.0
                    v2 = v2o / v2u
                    v_12 = v1 * v2
                    sum_v_12 = v_12.sum()
                    v_xy[idx] = (
                        (4.0 * self.force.F * omega)
                        / (self.mat.rho * self.geo.a * self.geo.b * self.geo.h)
                    ) * sum_v_12

                v[idx_f] = v_xy

            if verbose:
                progress_percent = (idx_f + 1) / n_frq * 100
                if int(progress_percent) % 10 == 0:
                    if int(progress_percent) != last_print:
                        print(f"   {int(progress_percent)}%", end="")
                        last_print = int(progress_percent)
                    if last_print == 100:
                        print("\n")

        return v

class PlateParameterIndex:
    #width is at x axis
    #TODO check if switching both is right
    WIDTH = 1 #a
    HEIGHT = 0 #b

    THICKNESS = 2 #t or h
    LOSS_FACTOR = 3 #eta
    BOUNDARY_CONDITION = 4

    #excitation relative x position (jan's y position)
    FORCE_X = 6 
    #excitation relative y position (jan's x position)
    FORCE_Y = 5 

    YOUNGS_MODULUS = 7 #E
    POISSONS_RATIO = 8 #n_y
    DENSITY = 9 #rho

DEFAULT_PLATE_PARAMETERS = [
    0.9,#width
    0.6,#height
    0.003,#thickness
    0.02,#loss factor
    0,#boundary condition
    0.36/0.9,#force relative position x
    0.255/0.6, #force relative position y
    7e10,#youngs modulus N/m^2
    0.3, #poissons ratio
    2700. #density kg/m^3

]

@tensorclass
class PlateParameter:
    width : torch.Tensor
    height : torch.Tensor
    thickness : torch.Tensor
    loss_factor: torch.Tensor
    boundary_condition : torch.Tensor
    force_x : torch.Tensor
    force_y : torch.Tensor
    youngs_modulus : torch.Tensor
    poissons_ratio : torch.Tensor
    density: torch.Tensor

    @staticmethod
    def from_array(param_array):
        return  PlateParameter(
            width=param_array[...,PlateParameterIndex.WIDTH],
            height=param_array[...,PlateParameterIndex.HEIGHT],
            thickness=param_array[...,PlateParameterIndex.THICKNESS],
            loss_factor=param_array[...,PlateParameterIndex.LOSS_FACTOR],
            boundary_condition=param_array[...,PlateParameterIndex.BOUNDARY_CONDITION],
            force_x=param_array[...,PlateParameterIndex.FORCE_X],
            force_y=param_array[...,PlateParameterIndex.FORCE_Y],
            youngs_modulus=param_array[...,PlateParameterIndex.YOUNGS_MODULUS],
            poissons_ratio = param_array[...,PlateParameterIndex.POISSONS_RATIO],
            density = param_array[...,PlateParameterIndex.DENSITY],
            batch_size = param_array.shape[:-1],
        )
    

    def element_size(self):
        return element_size(self)

    def __format__(self, format_spec):
        return 'PlateParameter'

def tr_kirchhoff_velocity_field(
    params: PlateParameter,
    freq: torch.tensor,
    freq_max: int,
    node_count: int,
):
    """
    params: 
        shape: B x 8
    
    freq_max:
        how many eigenfrequency should be considered
    
    node_count:
        the discretization in x and y
        
    freq:
        shape: freqs
    
    result:
        shape: B x freqs x node_count x node_count
    """

    # Target dimensions: B x freqs  x node_count x node_count x freq_max x freq_max
    E = params.youngs_modulus[:, None, None, None, None, None]
    n_y = params.poissons_ratio[:, None, None, None, None, None]
    rho = params.density[:, None, None, None, None, None]
    eta = params.loss_factor[:, None, None, None, None, None]

    a = params.width[:, None, None, None, None, None]
    b = params.height[:, None, None, None, None, None]
    h = params.thickness[:, None, None, None, None, None]
    xF = params.force_x[:, None, None, None, None, None]
    yF = params.force_y[:, None, None, None, None, None]

    pi = math.pi

    device = E.device
    dtype = E.dtype

    #bending stiffness
    B = (E * h**3.0) / (12.0 * (1.0 - n_y**2.0))

    x_1 = torch.linspace(
        0.5/node_count,
        1-(0.5/node_count),
        node_count,
        device=E.device,
        dtype=E.dtype,
    )[None, None, :, None, None, None]

    x_2 = torch.linspace(
        0.5/node_count,
        1-(0.5/node_count),
        node_count,
        device=device,
        dtype=dtype,
    )[None, None, None, :, None, None]

    x_1 = x_1 * a
    x_2 = x_2 * b


    Omega = 2.0 * pi * freq[:,:,None,None,None,None]

    m = torch.arange(1, freq_max+1,dtype=dtype,device=device)
    m = m[None, None,None, None, :, None]
    n = torch.arange(1, freq_max+1,dtype=dtype,device=device)
    n = n[None, None,None, None, None, :]

    omega_mn = pi**2 * ((m / a) ** 2 + (n / b) ** 2) * torch.sqrt(B / (rho * h))


    v2o = omega_mn**2.0 - Omega**2.0 - 1.j * eta * omega_mn**2.0

    v2u = (omega_mn**2.0 - Omega**2.0) ** 2.0 + eta**2.0 * omega_mn**4.0
    v = v2o / v2u * torch.sin(pi*m*xF)*torch.sin(pi*n*yF)\
        * torch.sin(pi*m*x_1/a)*torch.sin(pi*n*x_2/b) \

    v = 4.*Omega/(rho*a*b*h)*v

    velocity_field = v.sum(dim=(-1,-2))
    return velocity_field


def tr_velocity_field_to_frequency_response(velocity_field,a=1,b=1,db_ref=1e-9,normalization=False):
    """
    velocity_field shape: B x freqs x node_count x node_count
    a shape: B
    b shape: B

    result:
        shape: B x freqs
    """
    if(torch.is_tensor(a)):
        a = a[:,None]
    if(torch.is_tensor(b)):
        b = b[:,None,None]
    velocity_field = torch.square(torch.absolute(velocity_field))
    velocity_field = velocity_field.mean(dim=-1)*b
    velocity_field = velocity_field.mean(dim=-1)*a

    pre_factor = 10.
    if(normalization):
        pre_factor/= FRF_STD 


    fr_func = pre_factor*(torch.log10(velocity_field+1e-12)-np.log10(db_ref))
    return fr_func




