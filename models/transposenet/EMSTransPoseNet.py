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
        self.reg_hidden_dim = config.get('reg_hidden_dim')
        self.hyper_dim = config.get('hyper_dim')
        self.hypernet_t_fc_h = nn.Linear(decoder_dim, self.reg_hidden_dim * (decoder_dim + 1))
        self.hypernet_t_fc_o = nn.Linear(decoder_dim, 3 * (self.reg_hidden_dim + 1))
        self.hypernet_rot_fc_h = nn.Linear(decoder_dim, self.reg_hidden_dim * (decoder_dim + 1))
        self.hypernet_rot_fc_o = nn.Linear(decoder_dim, 4 * (self.reg_hidden_dim + 1))

        # =========================================
        # Regressor Heads
        # =========================================
        self.hyper_regressor_head_t = HyperPoseRegressor(input_dim=decoder_dim,
                                                         hidden_dim=self.reg_hidden_dim,
                                                         output_dim=3)
        self.hyper_regressor_head_rot = HyperPoseRegressor(input_dim=decoder_dim,
                                                           hidden_dim=self.reg_hidden_dim,
                                                           output_dim=4)

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

        ##################################################
        # Hypernet
        ##################################################
        hyper_w_t_fc_h = self.hypernet_t_fc_h(global_desc_t)
        hyper_w_t_fc_o = self.hypernet_t_fc_o(global_desc_t)
        hyper_w_rot_h = self.hypernet_rot_fc_h(global_desc_rot)
        hyper_w_rot_o = self.hypernet_rot_fc_o(global_desc_rot)

        ##################################################
        # Regression
        ##################################################
        x_t = self.hyper_regressor_head_t(global_desc_t, hyper_w_t_fc_h, hyper_w_t_fc_o)
        x_rot = self.hyper_regressor_head_rot(global_desc_rot, hyper_w_rot_h, hyper_w_rot_o)

        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return expected_pose, scene_log_distr


class HyperPoseRegressor(nn.Module):
    """ An MLP with hypernetwork's weights to regress a pose component"""

    def __init__(self, input_dim, hidden_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        hidden_dim: (int) the hidden dimension
        output_dim: (int) the output dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self.fc_h = nn.Linear(self._input_dim, self._hidden_dim)

        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(self._input_dim * 2, self._hidden_dim)

        self.fc_o = nn.Linear(self._hidden_dim, self._output_dim)
        self._reset_parameters()

    @staticmethod
    def batched_linear_layer(x, wb):
        # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
        one = torch.ones(*x.shape[:-1], 1, device=x.device)
        linear_res = torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb)
        return linear_res.squeeze(1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, hyper_weights_h, hyper_weights_o):
        """
        Forward pass
        """
        x_hyper_h = F.gelu(self.batched_linear_layer(x, hyper_weights_h.view(hyper_weights_h.shape[0],
                                                                                   (self._input_dim + 1),
                                                                                   self._hidden_dim)))
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        x = torch.add(x, x_hyper_h)

        x_hyper_o = F.gelu(self.batched_linear_layer(x, hyper_weights_o.view(hyper_weights_o.shape[0],
                                                                             (self._hidden_dim + 1),
                                                                             self._output_dim)))
        x = self.fc_o(x)
        x = torch.add(x, x_hyper_o)
        return x
