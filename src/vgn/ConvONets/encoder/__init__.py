from vgn.ConvONets.encoder import (
    pointnet, local_dgcnn, local_vn_dgcnn, voxels, pointnetpp
)


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'dgcnn_local_pool': local_dgcnn.LocalPoolDGCNN,
    'vn_dgcnn_local_pool': local_vn_dgcnn.LocalPool_VN_DGCNN,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
}
