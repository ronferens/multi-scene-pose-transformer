"""
The Efficient Multi-Scene TransPoseNet model
"""

import torch
from .MSTransPoseNet import MSTransPoseNet, PoseRegressor


class EMSTransPoseNet(MSTransPoseNet):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__(config, pretrained_path)

        decoder_dim = self.transformer_t.d_model
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)

    def forward_heads(self, global_desc_t, global_desc_rot, scene_log_distr, max_indices):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        x_t = self.regressor_head_t(global_desc_t)
        x_rot = self.regressor_head_rot(global_desc_rot)
        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return expected_pose, scene_log_distr
