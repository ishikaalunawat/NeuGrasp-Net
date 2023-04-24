import torch
import torch.nn as nn
import torch.nn.functional as F
from vgn.ConvONets.conv_onet.models.utils.dgcnn_util import get_graph_feature
from torch_scatter import scatter_mean, scatter_max
from vgn.ConvONets.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from vgn.ConvONets.encoder.unet import UNet
from vgn.ConvONets.encoder.unet3d import UNet3D

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class LocalPoolDGCNN(nn.Module):
    ''' DGCNN encoder (point-wise) with projection to local plane/grid features
        Number of input points are fixed.
        Points get mean pooled locally to build the grid/plane features
        We have the option to use a dynamic or static graph
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        n_knn (int): number of neighbors for knn graph
        use_dg (bool): weather to use a dynamic graph (like dgcnn) or a static knn graph
        use_bnorm (bool): weather to use batch normalization
        scatter_type (str): feature aggregation when doing pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, n_knn=10, use_dg=False, use_bnorm=False,
                 scatter_type='mean', unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type=['xz', 'xy', 'yz'], padding=0.1):
        super(LocalPoolDGCNN, self).__init__()
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.n_knn = n_knn
        self.use_dg = use_dg

        if use_bnorm:
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            self.bn3 = nn.BatchNorm2d(hidden_dim)
            self.bn4 = nn.BatchNorm2d(hidden_dim)
            self.bn5 = nn.BatchNorm1d(c_dim)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
            self.bn5 = nn.Identity()
        
        self.pool1 = meanpool
        self.pool2 = meanpool
        self.pool3 = meanpool
        self.pool4 = meanpool
        
        self.conv1 = nn.Sequential(nn.Conv2d(dim*2, hidden_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(hidden_dim*4, c_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'mean':
            self.scatter = scatter_mean
        elif scatter_type == 'max':
            self.scatter = scatter_max
        else:
            raise ValueError('incorrect scatter type')
    
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = self.scatter(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = self.scatter(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid


    def forward(self, p):
        batch_size, N, D = p.size()

        x = p.clone().transpose(1, 2) # dgcnn needs B x D x N
        x, knn_idx = get_graph_feature(x, k=self.n_knn, return_idx=True)
        x = self.conv1(x)
        x1 = self.pool1(x)

        if self.use_dg:
            knn_idx = None # dynamic graph
        else:
            pass # keep using the first knn graph

        x = get_graph_feature(x1, k=self.n_knn, idx=knn_idx)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn, idx=knn_idx)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.n_knn, idx=knn_idx)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        # latent features per-point
        c = x.transpose(1, 2) # B x N x C

        # project features to planes/grid
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea