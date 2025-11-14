import torch
from torch import nn
from vibromodes.globals import AMPLITUDE_STD,AMPLITUDE_MEANS
from vibromodes.kirchhoff import PlateParameter
from vibromodes.models.nn import  DoubleConv, Down, Up, SelfAttention, Film
from vibromodes.velocity_field import field_ln_mag_dir2dict
from vibromodes.utils import convert_1d_to_interpolator




class FQOEncoder(nn.Module):
    def __init__(self,c_in=1,conditional=False,scaling_factor=32,len_conditional=None,gap_film=True):
        super().__init__()

        k = scaling_factor

        self.inc = DoubleConv(c_in, 2 * k)
        self.down0 = nn.Sequential(nn.Conv2d(2 * k, 2 * k, 3, stride=2, padding=1), nn.ReLU())

        self.down1 = Down(2 * k, 4 * k)
        self.down2 = Down(4 * k, 6 * k)
        self.sa2 = SelfAttention(6 * k)
        self.down3 = Down(6 * k, 8 * k)
        self.sa3 = SelfAttention(8 * k)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(len_conditional, 6 * k)
        self.bot1 = DoubleConv(8 * k, 8 * k)
        self.bot3 = DoubleConv(8 * k, 6 * k)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_film = Film(6 * k, 6 * k)

    
    def forward(self,x,conditional):
        x = torch.nn.functional.interpolate(x, size=(96, 128), mode='bilinear', align_corners=True)
        x = self.inc(x)
        x1 = self.down0(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x3 = self.sa2(x3)
        if self.conditional is True and conditional is not None:
            x3 = self.film(x3, conditional)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot3(x4)
        gap = self.global_avg_pool(x4)[:, :, 0, 0]
        x4 = self.gap_film(x4, gap)
        return x1, x2, x3, x4

class FQODecoder(nn.Module):
    def __init__(self,scaling_factor,out_channels=1,output_size=(61,91)):
        super().__init__()

        k = scaling_factor

        self.queryfilm1 = Film(1, 6 * k)

        self.up_project1 = DoubleConv(6 * k, 2 * k)
        self.up1 = Up(8 * k, 4 * k)
        self.sa4 = SelfAttention(4 * k)
        self.queryfilm2 = Film(1, 4 * k)

        self.up_project2 = DoubleConv(4 * k, 2 * k)
        self.up2 = Up(6 * k, 3 * k)
        self.queryfilm3 = Film(1, 3 * k)

        self.up_project3 = DoubleConv(2 * k, 1 * k)
        self.up3 = Up(4 * k, 2 * k)
        self.outc = nn.Conv2d(2 * k, out_channels, kernel_size=1)

        self.out_channels = out_channels
        self.output_size = output_size

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    
    def forward_features(self,xs,frequencies):
        x1,x2,x3,x4 = xs
        B, n_freqs = frequencies.shape

        x4 = x4.repeat_interleave(n_freqs, dim=0)
        x4 = self.queryfilm1(x4, frequencies.view(-1, 1))
        x = self.up1(x4, self.up_project1(x3).repeat_interleave(n_freqs, dim=0))
        x = self.sa4(x)
        x = self.queryfilm2(x, frequencies.view(-1, 1))
        x = self.up2(x, self.up_project2(x2).repeat_interleave(n_freqs, dim=0))
        readout = x
        x = self.queryfilm3(x, frequencies.view(-1, 1))
        x1_up = self.up_project3(x1).repeat_interleave(n_freqs, dim=0)
        #readout = torch.cat([x1_up, self.up_sample(x)], dim=1)
        x = self.up3(x, x1_up)

        output = self.outc(x)
        output = torch.nn.functional.interpolate(output, size=self.output_size, mode='bilinear', align_corners=True)
        
        return output.reshape(B, n_freqs,self.out_channels, output.size(2), output.size(3)),readout
    
    def forward(self,xs,frequencies):
        output,_ = self.forward_features(xs,frequencies)
        return output
    

class FQO_Unet(nn.Module):
    def __init__(self, conditional=False, scaling_factor=32, len_conditional=None,output_size=(61,91)):
        super().__init__()
        k = scaling_factor

        self.encoder = FQOEncoder(1,conditional,scaling_factor,len_conditional)
        self.decoder =FQODecoder(scaling_factor,3,output_size)
        #TODO does this make a difference?
        #torch.nn.init.normal_(self.decoder.outc.weight,mean=0,std=1e-3)

        self.nfe_counter = 0

        self.bias = convert_1d_to_interpolator((torch.tensor(AMPLITUDE_MEANS).float()/AMPLITUDE_STD),-1,1.)


    def reset_nfe_counter(self):
        self.nfe_counter = 0


    def forward(self, x, conditional:PlateParameter, frequencies):
        if(self.training):
            pass
    

        conditional = torch.stack([conditional["force_x"],conditional["force_y"],conditional["boundary_condition"]],dim=1)


        x = x.unsqueeze(1)
        xs = self.encoder(x, conditional)
        vel_fields = self.decoder(xs,frequencies)

        field =  vel_fields[:,:,0]

        bias = self.bias[frequencies]
        #bias of shape B x F
        field = field + bias[:,:,None,None]

        #normalization
        #vel_fields/=200.
        dir = vel_fields[:,:,[1,2]]

        return field_ln_mag_dir2dict({"ln_mag": field,
                                      "dir":dir}),{}




def get_net(conditional=False, len_conditional=None, scaling_factor=32):
    return FQO_Unet(conditional=conditional, len_conditional=len_conditional, scaling_factor=scaling_factor)
