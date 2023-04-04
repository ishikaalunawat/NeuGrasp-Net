import torch
import torch.nn as nn
from torch import distributions as dist
from vgn.ConvONets.conv_onet.models import decoder
from .pointnet_cls import PointNet

# Decoder dictionary
decoder_dict = {
    'simple_fc': decoder.FCDecoder,
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder,
    'picked_points': decoder.PickedPointDecoder,
}

class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()
        
        self.decoder_qual = decoders[0].to(device)
        # self.decoder_rot = decoders[1].to(device)
        self.decoder_width = decoders[1].to(device)
        if len(decoders) == 3:
            self.decoder_tsdf = decoders[2].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf

    def forward(self, inputs, grasp_query, p_tsdf=None, sample=True, **kwargs): # <- Changed to predict only grasp quality
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        #p, _ = grasp_query # <- Changed to predict only grasp quality
        # if isinstance(grasp_query, tuple):
        #     query, _ = grasp_query
        # else:
        #     query = grasp_query

        # if isinstance(p, dict):
        #     batch_size = p['p'].size(0)
        # else:
        #     batch_size = p.size(0)

        c = self.encode_inputs(inputs)
        # feature = self.query_feature(p, c)
        # qual, rot, width = self.decode_feature(p, feature)
        qual, width = self.decode(grasp_query, c) # <- Changed to predict only grasp quality
        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            return qual, width, tsdf # <- Changed to predict only grasp quality
        else:
            return qual, width # <- Changed to predict only grasp quality
            
    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        #rot = self.decoder_rot(p, c, **kwargs)
        #rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, width

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = - qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            #print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, width_out

class ConvolutionalOccupancyNetworkGeometry(nn.Module):
    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        
        self.decoder_tsdf = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, p, p_tsdf, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf
    
    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c
        
    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

class PointNetGPD(nn.Module):
    def __init__(self, dim=7, c_dim=32,
                 out_dim=1,
                 point_network='pointnet',
                 sample_mode='bilinear', 
                 padding=0.1,
                 concat_feat=True):
        super().__init__()
        
        self.dim = dim # input
        self.out_dim = out_dim
        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
            c_dim += 3 # since we also append local grasp pc
        self.c_dim = c_dim
        self.sample_mode = sample_mode
        self.padding = padding

        self.fc_g = nn.Linear(dim, c_dim) # Linear layer to encode input grasp center and orientation
        if point_network == 'pointnet':
            self.point_network = PointNet(input_dim=c_dim, num_class=out_dim)


    def forward(self, grasp_query):
        if isinstance(grasp_query, tuple):
            pos, rotations, grasps_pc_local, _ = grasp_query
            # zero_pc_indices = grasps_pc.sum(dim=2) == 0
            f = torch.cat([pos,rotations], dim = 2) # <- Changed to predict only grasp quality
            c = []
        else:
            raise NotImplementedError

        # plane_type = list(c_plane.keys())
        # if self.concat_feat:
        #     c = []
        #     if 'xz' in plane_type:
        #         c.append(self.sample_plane_feature(grasps_pc, c_plane['xz'], plane='xz'))
        #     if 'xy' in plane_type:
        #         c.append(self.sample_plane_feature(grasps_pc, c_plane['xy'], plane='xy'))
        #     if 'yz' in plane_type:
        #         c.append(self.sample_plane_feature(grasps_pc, c_plane['yz'], plane='yz'))
        #     c = torch.cat(c, dim=1)
        #     c = c.transpose(1, 2)
        #     # zero the features for zero points in the grasp_pc
        #     c[zero_pc_indices] *= torch.tensor(0, dtype=c.dtype, device=c.device)
        # else:
        #     raise NotImplementedError
        # concat grasp point cloud in local frame
        # c = torch.cat([grasps_pc_local, c], dim=2)
        c = grasps_pc_local
        
        # Linear layer to encode input grasp center and orientation
        g = self.fc_g(f)
        print(g.size(), c.size())
        queries = torch.cat([g, c], dim=1)
        
        queries = queries.transpose(2, 1) # Transpose to get shape B, D, N
        out = self.point_network(queries)

        return out