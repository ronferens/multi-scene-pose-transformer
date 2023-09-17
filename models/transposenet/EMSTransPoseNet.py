"""
The Efficient Multi-Scene TransPoseNet model
"""

import torch
from torch import nn
import torch.nn.functional as F
from .MSTransPoseNet import MSTransPoseNet, PoseRegressor


class EMSTransPoseNet(MSTransPoseNet):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__(config, pretrained_path)

        decoder_dim = self.transformer_t.d_model

        # =========================================
        # Hypernetwork
        # =========================================
        self.hyper_dim_t = config.get('hyper_dim_t')
        self.hyper_in_t_proj = nn.Linear(in_features=1280, out_features=decoder_dim)
        self.hyper_in_t_fc_0 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_t_fc_1 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_t_fc_2 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hypernet_t_fc_h0 = nn.Linear(decoder_dim, self.hyper_dim_t * (decoder_dim + 1))
        self.hypernet_t_fc_h1 = nn.Linear(decoder_dim, (self.hyper_dim_t // 2) * (self.hyper_dim_t + 1))
        self.hypernet_t_fc_h2 = nn.Linear(decoder_dim, 3 * ((self.hyper_dim_t // 2) + 1))

        self.hyper_dim_rot = config.get('hyper_dim_rot')
        self.hyper_in_rot_proj = nn.Linear(in_features=1280, out_features=decoder_dim)
        self.hyper_in_rot_fc_0 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_rot_fc_1 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_rot_fc_2 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hypernet_rot_fc_h0 = nn.Linear(decoder_dim, self.hyper_dim_rot * (decoder_dim + 1))
        self.hypernet_rot_fc_h1 = nn.Linear(decoder_dim, self.hyper_dim_rot * (self.hyper_dim_rot + 1))
        self.hypernet_rot_fc_h2 = nn.Linear(decoder_dim, 4 * (self.hyper_dim_rot + 1))

        # =========================================
        # Regressor Heads
        # =========================================
        # (1) Hypernetworks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(decoder_dim, self.hyper_dim_t, 3, hidden_scale=0.5)
        self.regressor_hyper_rot = PoseRegressorHyper(decoder_dim, self.hyper_dim_rot, 4, hidden_scale=1.0)

        # (2) Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)

        self.w_t, self.w_rot = None, None

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    def forward_heads(self):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """

        ##################################################
        # Hypernet
        ##################################################
        t_input = torch.add(self._global_desc_t, self.hyper_in_t_proj(self._embeds))
        hyper_in_h0 = self._swish(self.hyper_in_t_fc_0(t_input))
        hyper_w_t_fc_h0 = self.hypernet_t_fc_h0(hyper_in_h0)
        hyper_in_h1 = self._swish(self.hyper_in_t_fc_1(t_input))
        hyper_w_t_fc_h1 = self.hypernet_t_fc_h1(hyper_in_h1)
        hyper_in_h2 = self._swish(self.hyper_in_t_fc_2(t_input))
        hyper_w_t_fc_h2 = self.hypernet_t_fc_h2(hyper_in_h2)

        rot_input = torch.add(self._global_desc_rot, self.hyper_in_rot_proj(self._embeds))
        hyper_in_h0 = self._swish(self.hyper_in_rot_fc_0(rot_input))
        hyper_w_rot_fc_h0 = self.hypernet_rot_fc_h0(hyper_in_h0)
        hyper_in_h1 = self._swish(self.hyper_in_rot_fc_1(rot_input))
        hyper_w_rot_fc_h1 = self.hypernet_rot_fc_h1(hyper_in_h1)
        hyper_in_h2 = self._swish(self.hyper_in_rot_fc_2(rot_input))
        hyper_w_rot_fc_h2 = self.hypernet_rot_fc_h2(hyper_in_h2)

        self.w_t = {'w_h1': hyper_w_t_fc_h0, 'w_h2': hyper_w_t_fc_h1, 'w_o': hyper_w_t_fc_h2}
        self.w_rot = {'w_h1': hyper_w_rot_fc_h0, 'w_h2': hyper_w_rot_fc_h1, 'w_o': hyper_w_rot_fc_h2}

        ##################################################
        # Regression
        ##################################################
        # (1) Hypernetwork's regressors
        x_hyper_t = self.regressor_hyper_t(self._global_desc_t, self.w_t)
        x_hyper_rot = self.regressor_hyper_rot(self._global_desc_rot, self.w_rot)

        # (2) Trained regressors
        x_t = self.regressor_head_t(self._global_desc_t)
        x_rot = self.regressor_head_rot(self._global_desc_rot)

        ##################################################
        # Output
        ##################################################
        x_t = torch.add(x_t, x_hyper_t)
        x_rot = torch.add(x_rot, x_hyper_rot)

        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return expected_pose, self._scene_log_distr


class PoseRegressorHyper(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, hidden_dim, output_dim, hidden_scale=1.0):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension
        hidden_scale: (float) Ratio between the input and the hidden layers' dimensions
        """
        super().__init__()
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_scale = hidden_scale

    @staticmethod
    def batched_linear_layer(x, wb):
        # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
        one = torch.ones(*x.shape[:-1], 1, device=x.device)
        linear_res = torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb)
        return linear_res.squeeze(1)

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    def forward(self, x, weights):
        """
        Forward pass
        """
        x = self._swish(self.batched_linear_layer(x, weights.get('w_h1').view(weights.get('w_h1').shape[0],
                                                                              (self.decoder_dim + 1),
                                                                              self.hidden_dim)))
        for index in range(len(weights.keys()) - 2):
            x = self._swish(self.batched_linear_layer(x,
                                                      weights.get(f'w_h{index + 2}').view(
                                                          weights.get(f'w_h{index + 2}').shape[0],
                                                          (self.hidden_dim + 1),
                                                          (int(self.hidden_dim * self.hidden_scale)))))
        x = self.batched_linear_layer(x, weights.get('w_o').view(weights.get('w_o').shape[0],
                                                                 (int(self.hidden_dim * self.hidden_scale) + 1),
                                                                 self.output_dim))
        return x