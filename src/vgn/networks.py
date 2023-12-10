from builtins import super
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from vgn.ConvONets.conv_onet.config import get_model
from vgn.ConvONets.conv_onet.models import PointNetGPD

def get_network(name):
    models = {
        "vgn": ConvNet,
        "giga_aff": GIGAAff,
        "giga": GIGA,
        "giga_hr": GIGAHighRes,
        "giga_hr_deeper": GIGAHighResDeeper,
        "giga_hr_affnet": GIGAHighResAffNet,
        "giga_geo": GIGAGeo,
        "giga_detach": GIGADetach,
        "neu_grasp_pn": NeuGraspPN,
        "neu_grasp_pn_deeper": NeuGraspPNDeeper,
        "neu_grasp_pn_deeper2": NeuGraspPNDeeper2,
        "neu_grasp_pn_deeper3": NeuGraspPNDeeper3,
        "neu_grasp_pn_deeper4": NeuGraspPNDeeper4,
        "neu_grasp_pn_deeper5": NeuGraspPNDeeper5,
        "neu_grasp_pn_affnet": NeuGraspPNAffNet,
        "neu_grasp_pn_affnet_deeper": NeuGraspPNAffNetDeeper,
        "neu_grasp_pn_affnet_sem": NeuGraspPNAffNetSem,
        "neu_grasp_pn_affnet_sem_deeper": NeuGraspPNAffNetSemDeeper,
        "neu_grasp_pn_detach": NeuGraspPNDetach,
        "neu_grasp_pn_no_local_cloud": NeuGraspPNNoLocalCloud,
        "neu_grasp_pn_pn": NeuGraspPNPN,
        "neu_grasp_pn_pn_deeper": NeuGraspPNPNDeeper,
        "neu_grasp_vn_pn_pn": NeuGraspVNPNPN,
        "neu_grasp_vn_pn_pn_deeper": NeuGraspVNPNPNDeeper,
        "neu_grasp_dgcnn": NeuGraspDGCNN,
        "neu_grasp_dgcnn_deeper": NeuGraspDGCNNDeeper,
        "neu_grasp_dgcnn_no_local_cloud": NeuGraspDGCNNNoLocalCloud,
        "neu_grasp_dgcnn_pn": NeuGraspDGCNNPN,
        "neu_grasp_vn_dgcnn_pn": NeuGraspVNDGCNNPN,
        "pointnetgpd": PointNetGPD,
    }
    return models[name.lower()]()


def load_network(path, device, model_type=None):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    if model_type is None:
        model_name = '_'.join(path.stem.split("_")[1:-1])
    else:
        model_name = model_type
    print(f'Loading [{model_type}] model from {path}')
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
    )


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        qual_out = torch.sigmoid(self.conv_qual(x))
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)
        return qual_out, rot_out, width_out

def GIGAAff():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': False,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)

def GIGA():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7, # <- 3:7 Changed to predict only grasp quality
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def GIGAHighRes():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7, # <- 3:7 Changed to predict only grasp quality
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def GIGAHighResDeeper():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7, # <- 3:7 Changed to predict only grasp quality
            'sample_mode': 'bilinear',
            'hidden_size': 256,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_size': 256, 
    }
    return get_model(config)

def GIGAHighResAffNet():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'simple_local',
        'decoder_affrdnce': 'simple_local',
        'decoder_kwargs': {
            'dim': 7, # <- 3:7 Changed to predict only grasp quality
            'sample_mode': 'bilinear',
            'hidden_size': 128,
            'concat_feat': True
        },
        'decoder_tsdf': 'simple_local',
        'padding': 0,
        'c_dim': 64,
        'hidden_size': 128, 
    }
    return get_model(config)

def GIGAGeo():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': 'simple_local',
        'tsdf_only': True,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)

def GIGADetach():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': 'simple_local',
        'detach_tsdf': True,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)

def NeuGraspPN():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def NeuGraspPNDeeper():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNDeeper2():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 64,
        'hidden_dim': 128 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNDeeper3():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNDeeper4():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 64,
        'hidden_dim': 128 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNDeeper5():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNAffNet():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_affrdnce': 'picked_points',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True,
        },
        'decoder_tsdf': 'simple_local',
        'padding': 0,
        'c_dim': 64,
        'hidden_dim': 128 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNAffNetDeeper():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_affrdnce': 'picked_points',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True,
        },
        'decoder_tsdf': 'simple_local',
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNAffNetSem():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_affrdnce': 'picked_points',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True,
        },
        'decoder_tsdf': 'simple_local',
        'decoder_sem': 'simple_local',
        'padding': 0,
        'c_dim': 64,
        'hidden_dim': 128 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNAffNetSemDeeper():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_affrdnce': 'picked_points',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True,
        },
        'decoder_tsdf': 'simple_local',
        'decoder_sem': 'simple_local',
        'padding': 0,
        'c_dim': 64,
        'hidden_dim': 128 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspPNDetach():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'detach_tsdf': True,
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def NeuGraspPNNoLocalCloud():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'concat_local_cloud': False,
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def NeuGraspPNPN():
    config = {
        'encoder': 'pointnet_local_pool',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32,
        'hidden_dim': 128
    }
    return get_model(config)

def NeuGraspPNPNDeeper():
    config = {
        'encoder': 'pointnet_local_pool',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspDGCNN():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'dgcnn',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def NeuGraspDGCNNDeeper():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'dgcnn',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

def NeuGraspDGCNNNoLocalCloud():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'dgcnn',
            'concat_local_cloud': False,
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32 
    }
    return get_model(config)

def NeuGraspDGCNNPN():
    config = {
        'encoder': 'dgcnn_local_pool',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'dgcnn',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32,
        'hidden_dim': 128
    }
    return get_model(config)

def NeuGraspVNDGCNNPN():
    config = {
        'encoder': 'vn_dgcnn_local_pool',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'dgcnn',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32,
        'hidden_dim': 128
    }
    return get_model(config)

def NeuGraspVNPNPN():
    config = {
        'encoder': 'vn_pointnet_local_pool',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32,
        'hidden_dim': 128
    }
    return get_model(config)

def NeuGraspVNPNPNDeeper():
    config = {
        'encoder': 'vn_pointnet_local_pool',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 64,
            'unet': True,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 64
            }
        },
        'decoder': 'picked_points',
        'decoder_tsdf': 'simple_local',
        'decoder_kwargs': {
            'dim': 7,
            'point_network': 'pointnet',
            'sample_mode': 'bilinear',
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 128,
        'hidden_dim': 256 # for simple_local decoder
    }
    return get_model(config)

class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv(in_channels, filters[0], kernels[0])
        self.conv2 = conv(filters[0], filters[1], kernels[1])
        self.conv3 = conv(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = F.interpolate(x, 10)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.interpolate(x, 20)
        x = self.conv3(x)
        x = F.relu(x)

        x = F.interpolate(x, 40)
        return x


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx 
    
def count_num_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)