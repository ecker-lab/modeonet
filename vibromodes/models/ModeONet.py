from abc import abstractmethod
import math
import numpy as np
import torch
from torch import nn
from vibromodes.globals import AMPLITUDE_MEANS, AMPLITUDE_STD
from vibromodes.kirchhoff import PlateParameter
from vibromodes.models.FQOUNet import FQOEncoder,FQODecoder
from vibromodes.models.nn import MLP, DoubleConv,Film, SelfAttention, Up
from vibromodes.modes import complex_log_sum_dir, calc_mode_response_log_mag,calc_mode_response
from vibromodes.utils import convert_1d_to_interpolator
from vibromodes.velocity_field import field_ln_mag_dir2dict



ENABLE_BIAS = True




class CrossAttentionConditioning(nn.Module):
    def __init__(self,spatial_dim,temporal_dim,head_dim=32):
        super().__init__()
        self.spatial_norm = nn.GroupNorm(1,spatial_dim)
        self.temporal_norm = nn.LayerNorm(temporal_dim)

        self.film_temporal_on_spatial = Film(temporal_dim,spatial_dim)

        self.project_temporal_down = nn.Linear(temporal_dim,spatial_dim,bias=ENABLE_BIAS)
        self.project_temporal_up = nn.Linear(spatial_dim,temporal_dim,bias=ENABLE_BIAS)

        #per head there should be a dimension of 32
        num_heads = spatial_dim//head_dim
        assert spatial_dim%head_dim == 0

        self.attention = nn.MultiheadAttention(
            spatial_dim,num_heads,batch_first=True
        )

        #self.positional_embed = PositionalEncoding2D(spatial_dim)


    
    def forward(self,spatial_feat,temporal_feat):
        """
        spatial_feat shape: (B*n_peaks,s_dim,W,H)
        temporal_feat shape: (B*n_peaks,t_dim)
        result:
            spatial_feat shape: (B*n_peaks,s_dim,W,H)
            temporal_feat shape: (B*n_peaks,t_dim)
        
        """
        spatial_feat_skip = spatial_feat
        temporal_feat_skip = temporal_feat

        #pre norm
        spatial_feat = self.spatial_norm(spatial_feat)
        temporal_feat = self.temporal_norm(temporal_feat)

        temporal_project = self.project_temporal_down(temporal_feat)


        temporal_project = temporal_project.unsqueeze(1)
        #temporal_project of shape B*n_peaks,1,s_dim,


        spatial_feat_val = spatial_feat.permute(
            0,2,3,1
        )

        #spatial_feat_val shape: B*n_peaks,W,H,s_dim

        #pos_emb = self.positional_embed(spatial_feat_val)
        pos_emb = 0.
        spatial_feat_key = spatial_feat_val + pos_emb

        spatial_feat_val = spatial_feat_val.flatten(start_dim=1,end_dim=2)
        spatial_feat_key = spatial_feat_key.flatten(start_dim=1,end_dim=2)


        #pool
        temporal_project,attn_weight = self.attention(
            temporal_project, spatial_feat_key,spatial_feat_val
        )
        #temporal_project shape: B*n_peaks,1,s_dim
        temporal_project = temporal_project.squeeze(1)

        temporal_project = self.project_temporal_up(temporal_project)

        #conditioning
        spatial_feat = self.film_temporal_on_spatial(spatial_feat,temporal_feat)
        spatial_feat = spatial_feat + spatial_feat_skip
        #spatial_feat = spatial_feat_skip

        temporal_feat = temporal_project + temporal_feat_skip

        return spatial_feat,temporal_feat

class QueryConditioning(nn.Module):
    def __init__(self,n_peaks,dim,mode):
        super().__init__()
        self.embedding_bias = None
        self.embedding_scale = None
        self.film = None

        if "bias" in mode:
            self.embedding_bias = nn.Embedding(n_peaks,dim)
            nn.init.normal_(self.embedding_bias.weight,mean=0,std=1e-3)
        if "bias+scale" in mode:
            self.embedding_scale = nn.Embedding(n_peaks,dim)
        
        if "film" == mode:
            self.film = Film(1,dim)
        if not ( self.embedding_bias or self.embedding_scale or self.film):
            raise ValueError(f"Query conditioning mode {mode} is not supported!")

        self.n_peaks = n_peaks

    def forward_linear_con(self,x):
        x_shape = x.shape
        B_npeaks = x.shape[0]
        dim = x.shape[1]
        B = B_npeaks//self.n_peaks

        assert B*self.n_peaks == x.shape[0]

        x = x.view(B,self.n_peaks,dim,-1)
        #weight shape: n_peaks x dim

        if self.embedding_scale:
            weight = self.embedding_scale.weight
            x = x * weight[None,:,:,None]

        if self.embedding_bias:
            weight = self.embedding_bias.weight
            x = x + weight[None,:,:,None]
        
        x = x.view(*x_shape)
        return x

    def forward(self,x,queries):
        """
        x of shape (B*n_peaks,dim,...)
        queries: B*n_peaks,1
        """
        if self.embedding_bias or self.embedding_scale:
            x = self.forward_linear_con(x)

        if self.film:
            x = self.film(x,queries)
        
        return x



class ModeEmbeddingBlock(nn.Module):
    def __init__(self,spatial_dim_in,spatial_dim_out,
                 encoder_feat_dim,encoder_feat_project_dim,
                 temporal_dim,
                 spatial_self_attention=True,
                conditioning=CrossAttentionConditioning,
                head_dim = 32,
                n_peaks=16,
                query_conditioning_mode = "film"
                 ):
        """
        query_conditioning_mode: "film", "bias" ,"scale", "bias+scale"
        """
        super().__init__()
        self.spatial_dim_in = spatial_dim_in
        self.spatial_dim_out = spatial_dim_out
        self.encoder_feat_dim = encoder_feat_dim
        self.temporal_dim = temporal_dim

        self.spatial_query_emb = QueryConditioning(n_peaks,spatial_dim_in,query_conditioning_mode)
        self.temporal_query_emb = QueryConditioning(n_peaks,temporal_dim,query_conditioning_mode)

        self.temporal_norm1 = nn.LayerNorm(temporal_dim+encoder_feat_dim)
        #self.temporal_norm2 = nn.LayerNorm(temporal_dim)
        #TODO see if factor 4 has impact
        self.temporal_ffn = nn.Sequential(
            nn.Linear(temporal_dim+encoder_feat_dim,4*temporal_dim,bias=ENABLE_BIAS),
            nn.GELU(),
            nn.Linear(4*temporal_dim,temporal_dim,bias=ENABLE_BIAS)
        )


        self.up_project = DoubleConv(encoder_feat_dim,encoder_feat_project_dim)
        self.up = Up(encoder_feat_project_dim+spatial_dim_in,spatial_dim_out)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if spatial_self_attention:
            self.sa = SelfAttention(spatial_dim_out)
        else:
            self.sa = nn.Identity()
        
        self.conditioning = conditioning(spatial_dim_out,temporal_dim,head_dim)
        
        #self.spatial_norm = nn.BatchNorm2d(spatial_dim_out)

    def forward_temporal(self,temporal_feat,encoder_feat,mode_query):
        """

        temporal_feat shape: (B*n_peaks,t_dim)
        encoder_feat shape: (B,enc_dim,W,H)
        mode_query shape: (B*n_peaks)
        
        result:
            temporal_feat shape: (B*n_peaks,enc_dim)
        """

        n_peaks = temporal_feat.shape[0]//encoder_feat.shape[0]
        assert temporal_feat.shape[0] == encoder_feat.shape[0]*n_peaks

        temporal_skip = temporal_feat
        #temporal_feat = self.film_query_on_temporal(temporal_feat,mode_query)
        temporal_feat = self.temporal_query_emb(temporal_feat,mode_query)


        temporal_encoder_feat = self.pool(encoder_feat)
        temporal_encoder_feat = temporal_encoder_feat.flatten(start_dim=1)
        temporal_encoder_feat = temporal_encoder_feat.repeat_interleave(n_peaks,dim=0)
        temporal_feat = torch.cat([temporal_feat,temporal_encoder_feat],dim=1)

        temporal_feat = self.temporal_norm1(temporal_feat)
        temporal_feat = self.temporal_ffn(temporal_feat)
        temporal_feat = temporal_feat + temporal_skip
        return temporal_feat

    def forward_spatial(self,spatial_feat,encoder_feat,mode_query):
        """
        spatial_feat shape: (B*n_peaks,s_dim_in,W,H)
        encoder_feat shape: (B,enc_dim,W,H)
        mode_query shape: (B*n_peaks)

        result:
            spatial_feat shape: (B*n_peaks,s_dim_out,W*2,H*2)
        """
        n_peaks = spatial_feat.shape[0]//encoder_feat.shape[0]
        assert spatial_feat.shape[0] == encoder_feat.shape[0]*n_peaks

        spatial_feat = self.spatial_query_emb(spatial_feat,mode_query)

        spatial_feat = self.up(spatial_feat,
                self.up_project(encoder_feat).repeat_interleave(n_peaks,dim=0)
        )
        spatial_feat = self.sa(spatial_feat)
        return spatial_feat
    
    def forward(self,spatial_feat,temporal_feat,encoder_feat,mode_query):
        """processes the spatial and temporal features based on the encoder feats


        spatial_feat shape: (B*n_peaks,s_dim_in,W,H)
        temporal_feat shape: (B*n_peaks,t_dim)
        encoder_feat shape: (B,enc_dim,W,H)
        mode_query shape: (B*n_peaks)

        result:
            spatial_feat shape: (B*n_peaks,s_dim_out,W*2,H*2)
            temporal_feat shape: (B*n_peaks,enc_dim)
        """



        #temporal route
        temporal_feat = self.forward_temporal(temporal_feat,encoder_feat,mode_query)

        #spatial route 
        spatial_feat = self.forward_spatial(spatial_feat,encoder_feat,mode_query)

        spatial_feat,temporal_feat = self.conditioning(spatial_feat,temporal_feat)

        return spatial_feat,temporal_feat

def complex2log_mag_dir(x,epsilon=1e-8):
    """
    x of shape: B x ...
    and of dtype complex
    """
    log_mag = 0.5*torch.log(x.abs().square()+epsilon)
    x = x /torch.clip(x.abs(),1e-8)
    dir_real = torch.real(x)
    dir_imag = torch.imag(x)

    return torch.stack([log_mag,dir_real,dir_imag],dim=1)


class ModeShapeDecoder(nn.Module):
    #mode_shape_feat_dim = 2*k
    def __init__(self,mode_shape_feat_dim,out_c,n_peaks,output_size):
        super().__init__()

        self.out_c = out_c
        self.outc = nn.Conv2d(mode_shape_feat_dim, self.out_c, kernel_size=1)
        self.n_peaks = n_peaks
        self.output_size = output_size

    
    def forward(self,mode_shape_feat):
        """
        spatial_feat shape: B*n_peaks x K x W x H
        result:
            spatial part shape: B x out_c x n_peaks x W x H
        """
        assert mode_shape_feat.shape[0]%self.n_peaks == 0


        mode_shape_feat = self.outc(mode_shape_feat)
        mode_shape_feat = torch.nn.functional.interpolate(mode_shape_feat, size=self.output_size, mode='bilinear', align_corners=True)
        #spatial_feat shape: B*n_peaks x 2 x W x H


        B = mode_shape_feat.shape[0]//self.n_peaks
        _,_,W,H = mode_shape_feat.shape

        mode_shape_feat = mode_shape_feat.view(B,self.n_peaks,self.out_c,W,H)  
        mode_shape_feat = mode_shape_feat.permute(0,2,1,3,4)


        return mode_shape_feat

class ModeResponseFilmDecoder(nn.Module):
    def __init__(self,response_dim,out_c,n_peaks):
        super().__init__()
        self.out_c = out_c
        self.n_peaks = n_peaks


        self.freq_film = Film(1,response_dim)

        self.norm = nn.LayerNorm(response_dim)
        self.mlp = nn.Sequential(
            nn.Linear(response_dim,response_dim,bias=ENABLE_BIAS),
            nn.GELU(),
            nn.Linear(response_dim,self.out_c)
        )
    

    
    def forward(self,mode_response_feat,frequencies):
        """
        temporal_feat shape: B*n_peaks x temporal_dim
        frequencies: B x F
        result:
            shape B x F x out_c x n_peaks
        """
        B,F = frequencies.shape
        assert mode_response_feat.shape[0] == B*self.n_peaks

        frequencies = frequencies.repeat_interleave(self.n_peaks,dim=0)
        #frequencies_normed shape: B*n_peaks x F
        frequencies = frequencies.view(-1,1)
        #frequencies shape: B*n_peaks*F x 1
        mode_response_feat = mode_response_feat.repeat_interleave(F,dim=0)
        #temporal_feat shape: B*n_peaks*F x temporal_dim
        temporal_feat_skip = mode_response_feat
        mode_response_feat = self.freq_film(mode_response_feat,frequencies)


        mode_response_feat = self.norm(mode_response_feat)
        mode_response_feat = mode_response_feat +temporal_feat_skip


        #mode_response_feat = self.mlp(mode_response_feat)
        mode_response_feat = self.mlp(mode_response_feat)



        #temporal_feat shape: B*n_peaks*F x 2

        mode_response_feat = mode_response_feat.view(B,self.n_peaks,F,self.out_c)
        mode_response_feat = mode_response_feat.permute(0,2,3,1)
        return mode_response_feat


class ModeResponseAnalyticDecoder(nn.Module):

    def __init__(self,response_dim,n_peaks,linear_prediction):
        super().__init__()
        self.linear =nn.Linear(response_dim,response_dim)
        self.gelu = nn.GELU()
        self.out_height = nn.Linear(response_dim,
                                    2 if linear_prediction else 3
                                    )
        self.out_eigenfrequency = nn.Linear(response_dim,1)
        self.out_damping = nn.Linear(response_dim,1)
        self.linear_prediction = linear_prediction

        self.n_peaks = n_peaks
        means = torch.tensor(AMPLITUDE_MEANS)
        means -= means.mean()
        self.bias = convert_1d_to_interpolator(
            means.float(),0,1.)


    def forward(self,mode_response_feat,frequencies):
        """
        temporal_feat shape: B*n_peaks x temporal_dim
        frequencies: B x F
        result:
            shape B x F x out_c x n_peaks
        """
        B_npeaks, temporal_dim = mode_response_feat.shape
        B,F = frequencies.shape

        assert B_npeaks == B*self.n_peaks

        x = self.linear(mode_response_feat)
        x = self.gelu(x)
        
        eigen_freq = self.out_eigenfrequency(x)
        height = self.out_height(x)
        damping = self.out_damping(x)

        #scale frequencies between 0 and 1
        frequencies = (frequencies+1.)*0.5

        #make sure eigenfreq is positiv
        eigen_freq = nn.functional.softplus(eigen_freq*10.)
        #print(eigen_freq.min().item(),eigen_freq.max().item())

        #make sure damping is positiv 
        damping = nn.functional.softplus(damping)

        #the real damping factor is between 0.02
        #so divide by 10

        damping = damping/10.

        damping = damping.view(B,1,self.n_peaks)
        eigen_freq = eigen_freq.view(B,1,self.n_peaks)
        frequencies = frequencies.view(B,F,1)
        bias = self.bias[frequencies]

        if self.linear_prediction:
            height = height.view(B,1,self.n_peaks,2)
            response_r,response_i = calc_mode_response(frequencies,eigen_freq,damping)
            h_r = height[:,:,:,0]
            h_i = height[:,:,:,1]
            result_r = response_r*h_r - response_i*h_i
            result_i = response_r*h_i + response_i*h_r
            #result shape: B x F x n_peaks
            response = torch.stack([
                result_r,result_i 
            ],dim=2)
            #remove the bias since it will be added later
            response = response*torch.exp(-bias[:,:,:,None]*0.5)


            
        else: 
            height = height.view(B,1,self.n_peaks,3)
            response_mag, response_phase = calc_mode_response_log_mag(
                frequencies,eigen_freq,damping
            )
            response_mag = response_mag- 0.5*bias
            response_dir_x = torch.cos(response_phase)
            response_dir_y = torch.sin(response_phase)

            height_mag = height[:,:,:,0]
            result_mag = response_mag+height_mag

            height_dir_x = height[:,:,:,1]
            height_dir_y = height[:,:,:,2]
            norm = torch.linalg.norm(height[:,:,:,[1,2]],dim=-1)
            eps = torch.finfo(norm.dtype).eps
            norm = torch.clip(norm,eps)
            height_dir_x = height_dir_x/norm
            height_dir_y = height_dir_y/norm

            result_dir_x = response_dir_x*height_dir_x - response_dir_y*height_dir_y
            result_dir_y = response_dir_x*height_dir_y + response_dir_x*height_dir_y

            response = torch.stack([
                result_mag,result_dir_x,result_dir_y 
            ],dim=2)
            
        return response



            







class LogSuperposition(nn.Module):
    def __init__(self,correct_distribution):
        """
        correct_dist if true: 
            a bias is added such that the expected value is 0 
            and it is scaled such that the std is 1
        """
        super().__init__()
        self.correct_distribution = correct_distribution
        self.bias = convert_1d_to_interpolator(torch.tensor(AMPLITUDE_MEANS).float(),-1,1)
        pass

    @torch.autocast(device_type="cuda",enabled=False)
    def forward(self,mode_shapes,mode_responses,frequencies):
        """
        mode_shapes: 
            shape: B x 3 x n_peaks x W x H
            #not normalized

        mode_responses:
            shape: B x F x 3 x n_peaks
            #
        

        returns 
            velocity_field: log_vel_field
            aux_losses: 
        """
        B,_,n_peaks,W,H = mode_shapes.shape
        mode_shapes = mode_shapes.float()
        mode_responses = mode_responses.float()
        frequencies = frequencies.float()
        mode_shapes_ln_mag = self._normalize_mode(mode_shapes[:,0])
    
        #magnitude shape: B x F x n_peaks x W x H
        magnitude = mode_shapes_ln_mag[:,None,:,:,:]+mode_responses[:,:,0,:,None,None]
        s_r =  mode_shapes[:,None,1,:,:,:]
        t_r = mode_responses[:,:,1,:,None,None]



        s_i =  mode_shapes[:,None,2,:,:,:]
        t_i = mode_responses[:,:,2,:,None,None]


        #similar to cosine similarty stabilitation 
        eps = torch.finfo(s_r.dtype).eps
        assert s_r.dtype == s_i.dtype
        assert t_r.dtype == t_i.dtype

        s_norm = torch.sqrt(torch.clip(s_r**2+s_i**2,eps))
        s_r = s_r/s_norm
        s_i = s_i/s_norm

        t_norm = torch.sqrt(torch.clip(t_r**2+t_i**2,eps))
        t_r = t_r/t_norm
        t_i = t_i/t_norm

        real_dir = s_r*t_r - s_i*t_i
        imag_dir = s_r*t_i + s_i*t_r

        #sum[:,:,1] *=0
        #field_ln_mag = torch.logsumexp(sum[:,:,0],dim=2)

        #sum shape: B x F x 2 x n_peaks x W x H
        field_ln_mag, field_dir = complex_log_sum_dir(magnitude,real_dir,imag_dir,dim=2,epsilon=1e-3,stack_dim=-3)


        field_ln_mag = field_ln_mag - np.log(n_peaks)

        field_ln_mag = field_ln_mag*2



        field_ln_mag = field_ln_mag + self.bias[frequencies][:,:,None,None]
        field_ln_mag = field_ln_mag / AMPLITUDE_STD

        aux_loss = self.calc_aux_loss(mode_responses)
        B = mode_responses.shape[0]

        return field_ln_mag_dir2dict({"ln_mag":field_ln_mag,
                "dir":field_dir}),aux_loss



    def _normalize_mode(self,mode_shapes):
        """
        mode_shape shape: B x n_peaks x W x H

        """

        norm = 0.5*torch.logsumexp(2.*mode_shapes,dim=(-2,-1),keepdim=True)
        mode_shapes = mode_shapes - norm
        B,_,W,H = mode_shapes.shape
        if self.correct_distribution:

            #expected value of spatial_ln_mag = -0.5*np.sqrt(2*W*H)
            d = W*H
            expected_mean = - np.sqrt(2*np.log(d)) 
            mode_shapes = mode_shapes - expected_mean

        return mode_shapes


    def mode_response_normed2physical(self,mode_responses,frequencies):
        """
        mode_response shape: B x F x 3 x n_peaks
        result:
            B x F x n_peaks
        """
        n_peaks = mode_responses.shape[3]
        ln_mag = mode_responses[:,:,0] - np.log(n_peaks)
        ln_mag = ln_mag + self.bias[frequencies][:,:,None]*0.5

        mag = ln_mag.exp()
        dir_r = mode_responses[:,:,1]
        dir_i = mode_responses[:,:,2]

        norm = torch.sqrt(torch.clip(dir_r**2 + dir_i**2,1e-8))
        dir_r = dir_r/norm
        dir_i = dir_i/norm

        return dir_r*mag + dir_i*mag*1.j


    def mode_response_physical2normed(self,mode_responses,frequencies):
        """
        mode_responses shape: B x F x n_peaks
        frequencies: B x F
        result:
            shape: B x F x 3 x n_peaks
        """
        B,F,n_peaks = mode_responses.shape

        result = torch.empty(B,F,3,n_peaks,device=mode_responses.device,dtype=torch.float32)


        dir_r = torch.real(mode_responses)
        dir_i = torch.imag(mode_responses)

        norm = torch.sqrt(torch.clip(dir_r**2 + dir_i**2,1e-8))
        dir_r = dir_r/norm
        dir_i = dir_i/norm

        ln_mag = mode_responses.abs().log() + np.log(n_peaks) - self.bias[frequencies][:,:,None]*0.5

        result[:,:,0] = ln_mag
        result[:,:,1] = dir_r
        result[:,:,2] = dir_i

        return result
        


    def mode_shape_normed2physical(self,mode_shapes):
        """
        mode_shape:
            shape: B x 3 x n_peaks x W x H
        result:
            shape: B x n_peaks x W x H
        """
        ln_mag = mode_shapes[:,0]
        dir_r = mode_shapes[:,1]
        dir_i = mode_shapes[:,2]

        norm = torch.sqrt(torch.clip(dir_r**2 + dir_i**2,1e-8))
        dir_r = dir_r/norm
        dir_i = dir_i/norm

        mag = ln_mag.exp()

        return dir_r*mag + dir_i*mag*1.j


    def mode_shape_physical2normed(self,mode_shapes):
        """
        mode_shape:
            shape: B x n_peaks x W x H
        result:
            shape: B x 3 x n_peaks x W x H
        """
        B,n_peaks,W,H = mode_shapes.shape
        result = torch.empty(B,3,n_peaks,W,H,device = mode_shapes.device,
                             dtype = torch.float32)

        ln_mag = mode_shapes.abs().log()

        dir = mode_shapes/torch.clip(mode_shapes.abs(),1e-12)
        
        result[:,0]= ln_mag
        result[:,1] = torch.real(dir)
        result[:,2] = torch.imag(dir)
        return result

    def calc_aux_loss(self,mode_responses):
        """
        mode responses: B x F x out_c x n_peaks
        excitation_point: B x n_peaks
        """
        assert mode_responses.shape[2] == 3




        c = mode_responses[:,:,0]
        #c shape: B x F x n_peaks
        
        c = 0.5*torch.logsumexp(2.*c,dim=1)
        #c shape: B x F x n_peaks
        reg_loss = torch.nn.functional.softplus(c).mean()
        aux_sparse = reg_loss

        return {"aux_sparse":aux_sparse
                }


class ModeONet(nn.Module):
    def __init__(self,scaling_factor=32,n_peaks=12,
                 output_size=(61,91),linear_prediction=False,
                 temporal_dim = 128,query_conditioning_mode="film",
                 analytic_response = False,
                 ):
        super().__init__()
        k = scaling_factor
        self.encoder = FQOEncoder(conditional=True,scaling_factor=scaling_factor,
                                  len_conditional= 3,gap_film=False)

        queries = torch.linspace(-1,1,n_peaks).unsqueeze(0)
        self.register_buffer("queries",queries)
        self.bias = convert_1d_to_interpolator(torch.tensor(AMPLITUDE_MEANS).float()/AMPLITUDE_STD,-1,1.)

        self.n_peaks = n_peaks

        self.decoder_block1 = ModeEmbeddingBlock(
            6*k,4*k,6*k,2*k,temporal_dim,True,head_dim=k,
            query_conditioning_mode=query_conditioning_mode
        )

        self.decoder_block2 = ModeEmbeddingBlock(
            4*k,3*k,4*k,2*k,temporal_dim,False,head_dim=k,
            query_conditioning_mode=query_conditioning_mode,
        )

        self.decoder_block3 = ModeEmbeddingBlock(
            3*k,2*k,2*k,k,temporal_dim,False,head_dim=k,
            query_conditioning_mode=query_conditioning_mode
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.project_temporal_tokens = nn.Linear(6*k,temporal_dim,bias=ENABLE_BIAS)
        self.out_c = 3



        self.linear_prediction = linear_prediction


        self.temporal_norm1 = nn.LayerNorm(temporal_dim)

        outc = 2 if linear_prediction else 3
        if analytic_response:
            self.decode_temporal = ModeResponseAnalyticDecoder(temporal_dim,n_peaks,
                                                               linear_prediction=linear_prediction)
        else:
            self.decode_temporal = ModeResponseFilmDecoder(temporal_dim,outc,n_peaks)
        self.decode_spatial = ModeShapeDecoder(2*k,outc,self.n_peaks,output_size)
        if self.linear_prediction:
            self.superposition = LinearSuperposition(correct_distribution=True)
        else:
            self.superposition = LogSuperposition(correct_distribution=True)
        


    def forward_eigenmodes_mode_dynamics(self, pattern, phy_para:dict, frequencies):
        """
        result:
            spatial_part shape: B x 2 x n_peaks x W x H
            temporal_part shape: B x F x 2 x n_peaks
        """
    
        conditional = torch.stack([phy_para["force_x"],phy_para["force_y"],phy_para["boundary_condition"]],dim=1)


        pattern = pattern.unsqueeze(1)
        B = pattern.shape[0]


        encoder_feat = self.encoder(pattern, conditional)
        encoder_feat = list(encoder_feat)
        encoder_feat.reverse()


        queries = self.queries.repeat([B,1])

        queries = queries.view(-1,1)

        temporal_feat = self.pool(encoder_feat[0]).flatten(start_dim=1)
        temporal_feat = self.project_temporal_tokens(temporal_feat)
        temporal_feat = self.temporal_norm1(temporal_feat)
        temporal_feat = temporal_feat.repeat_interleave(self.n_peaks,dim=0)

        spatial_feat = encoder_feat[0].repeat_interleave(self.n_peaks,dim=0)
        spatial_feat,temporal_feat = self.decoder_block1(spatial_feat,temporal_feat,encoder_feat[1],queries)
        spatial_feat,temporal_feat = self.decoder_block2(spatial_feat,temporal_feat,encoder_feat[2],queries)
        spatial_feat,temporal_feat = self.decoder_block3(spatial_feat,temporal_feat,encoder_feat[3],queries)




        temporal_part = self.decode_temporal(temporal_feat,frequencies)
        spatial_part = self.decode_spatial(spatial_feat)

        #spatial part shape: B x 2 x n_peaks x W x H
        #temporal part shape: B x F x 2 x n_peaks


        return spatial_part,temporal_part
    



        

    
    def forward(self,x,conditional:PlateParameter,frequencies):
        spatial_part,temporal_part = self.forward_eigenmodes_mode_dynamics(x,conditional,frequencies)
        return self.superposition(spatial_part,temporal_part,frequencies)



        







class LinearSuperposition(nn.Module):
    def __init__(self,correct_distribution=False):
        super().__init__()
        self.bias = convert_1d_to_interpolator(torch.tensor(AMPLITUDE_MEANS).float(),-1,1.)

        self.eps = 1e-8
        self.correct_distribution = correct_distribution
    
    def mode_shape_physical2normed(self,x):
        """
        x of shape (B,n_peaks,W,H)

        result shape: 
            (B,2,n_peaks,W,H)
        """
        B,n_peaks,W,H = x.shape
        dtype = torch.float32 if x.dtype==torch.complex64 else torch.float64
        
        result = torch.empty(B,2,n_peaks,W,H,dtype=dtype)
        result[:,0] = torch.real(x)
        result[:,1] = torch.imag(x)
        return result
    
    def mode_response_physical2normed(self,x,freqs):
        """
        x of shape (B,F,n_peaks)

        result shape:
            (B,F,3,n_peaks)
        """
        B,F,n_peaks = x.shape

        bias = self.bias[freqs]
        factor = (-bias*0.5).exp()*n_peaks


        x = x * factor[:,:,None]

        result = torch.zeros(B,F,2,n_peaks)

        result[:,:,0] = torch.real(x)
        result[:,:,1] = torch.imag(x)

        return result

    

    def mode_shape_normed2physical(self,x):
        """
        x of shape (B,2,n_peaks,W,H)
        result shape:
            B,n_peaks,W,H
        """
        return x[:,0]+x[:,1]*1.j


    @abstractmethod
    def mode_response_normed2physical(self,x,freqs):
        """
        x of shape (B,F,2,n_peaks)

        result shape:
            B,F,n_peaks
        """
        B,F,_,n_peaks = x.shape
        bias = self.bias[freqs][:,:,None]
        factor = (bias*0.5).exp()/n_peaks
        
        return x[:,:,0]*factor + x[:,:,1]*factor*1.j
    
    def _normalize_mode(self,x):
        """
        (B,2,n_peaks,W,H)
        """
        B,_,n_peaks,W,H = x.shape
        tmp = x[:,[0,1]].permute(0,2,1,3,4)

        tmp = tmp.reshape(B,n_peaks,W*H*2)
        norm = torch.linalg.norm(tmp,dim=-1)

        #to have std of 1
        if self.correct_distribution:
            norm = norm/np.sqrt(2*W*H)
        
        x = x/(norm[:,None,:,None,None]+self.eps)

        return x
    

    def calc_aux_loss(self,mode_responses,frequencies):
        """
        mode_responses: (B,F,2,n_peaks)
        """
        (B,F,_,n_peaks) = mode_responses.shape

        mode_responses = mode_responses.reshape(B,F*2,n_peaks)

        mode_responses_norms = torch.linalg.norm(mode_responses,dim=1)
        result = torch.log(mode_responses_norms+1).mean(dim=1) 

        return {"aux_sparse":result.mean()}


        
    def forward(self,mode_shapes,mode_responses,freqs):
        """
        mode_shapes: (B,2,n_peaks,W,H)
        mode_responses: (B,F,2,n_peaks)
        freqs: (B,F)
        """
        mode_shapes = self._normalize_mode(mode_shapes)
        
        B,_,n_peaks,W,H = mode_shapes.shape
        _,F,_,_ = mode_responses.shape
        assert mode_shapes.shape == (B,2,n_peaks,W,H)
        assert mode_responses.shape == (B,F,2,n_peaks)
        
        #shape B x 1 x n_peaks x W x H
        s_r = mode_shapes[:,0].unsqueeze(1)
        s_i = mode_shapes[:,1].unsqueeze(1)

        #shape B x F x n_peaks x 1 x 1
        t_r = mode_responses[:,:,0].unsqueeze(-1).unsqueeze(-1)
        t_i = mode_responses[:,:,1].unsqueeze(-1).unsqueeze(-1)

        factor = 1./n_peaks

        r_new = (factor*s_r*t_r).sum(dim=2) - (factor*s_i*t_i).sum(dim=2)
        i_new = (factor*s_r*t_i).sum(dim=2) + (factor*s_i*t_r).sum(dim=2)


        ln_mag = torch.log(r_new**2 + i_new**2 + self.eps)
        ln_mag = ln_mag 
        ln_mag = ln_mag + self.bias[freqs][:,:,None,None]
        ln_mag = ln_mag/AMPLITUDE_STD

        norm = (r_new**2 + i_new**2).sum(dim=(-1,-2),keepdim=True)
        eps = torch.finfo(norm.dtype).eps
        norm = torch.clip(norm,eps)
        norm = torch.sqrt(norm)

        normed_field = torch.stack([r_new/norm,i_new/norm],dim=-3)

        aux_loss = self.calc_aux_loss(mode_responses,freqs)
        

        return {"ln_mag":ln_mag, "normed_field":normed_field},aux_loss


