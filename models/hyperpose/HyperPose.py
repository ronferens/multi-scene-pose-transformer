import torch
import torch.nn as nn
import torch.nn.functional as F
from .MSHyperPose import PoseRegressorHyper


class HyperPose(nn.Module):
    """
    An enhanced PoseNet implementation with Hyper-Network
    """
    def __init__(self, config, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(HyperPose, self).__init__()

        # Efficient net
        self.backbone = torch.load(backbone_path)
        backbone_dim = 1280
        latent_dim = 1024

        # Regressor layers
        self.fc1 = nn.Linear(backbone_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3)
        self.fc3 = nn.Linear(latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # =========================================
        # Hyper networks
        # =========================================
        self.hyper_dim_t = backbone_dim
        self.hyper_t_hidden_scale = config.get('hyper_t_hidden_scale')
        self.hyper_in_t_proj = nn.Linear(in_features=1280, out_features=self.hyper_dim_t)
        self.hyper_in_t_fc_0 = nn.Linear(in_features=self.hyper_dim_t, out_features=self.hyper_dim_t)
        self.hyper_in_t_fc_1 = nn.Linear(in_features=self.hyper_dim_t, out_features=self.hyper_dim_t)
        self.hyper_in_t_fc_2 = nn.Linear(in_features=self.hyper_dim_t, out_features=self.hyper_dim_t)
        self.hypernet_t_fc_h0 = nn.Linear(self.hyper_dim_t, self.hyper_dim_t * (self.hyper_dim_t + 1))
        self.hypernet_t_fc_h1 = nn.Linear(self.hyper_dim_t,
                                          int(self.hyper_dim_t * self.hyper_t_hidden_scale) * (self.hyper_dim_t + 1))
        self.hypernet_t_fc_h2 = nn.Linear(self.hyper_dim_t, 3 * (int(self.hyper_dim_t * self.hyper_t_hidden_scale) + 1))

        self.hyper_dim_rot = config.get('hyper_dim_rot')
        self.hyper_in_rot_proj = nn.Linear(in_features=1280, out_features=self.hyper_dim_rot)
        self.hyper_in_rot_fc_0 = nn.Linear(in_features=self.hyper_dim_rot, out_features=self.hyper_dim_rot)
        self.hyper_in_rot_fc_1 = nn.Linear(in_features=self.hyper_dim_rot, out_features=self.hyper_dim_rot)
        self.hyper_in_rot_fc_2 = nn.Linear(in_features=self.hyper_dim_rot, out_features=self.hyper_dim_rot)
        self.hypernet_rot_fc_h0 = nn.Linear(self.hyper_dim_rot, self.hyper_dim_rot * (self.hyper_dim_rot + 1))
        self.hypernet_rot_fc_h1 = nn.Linear(self.hyper_dim_rot, self.hyper_dim_rot * (self.hyper_dim_rot + 1))
        self.hypernet_rot_fc_h2 = nn.Linear(self.hyper_dim_rot, 4 * (self.hyper_dim_rot + 1))

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        # =========================================
        # Regressor Heads
        # =========================================
        # (1) Hyper-networks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(self.hyper_dim_t, self.hyper_dim_t, 3,
                                                    hidden_scale=self.hyper_t_hidden_scale)
        self.regressor_hyper_rot = PoseRegressorHyper(self.hyper_dim_rot, self.hyper_dim_rot, 4, hidden_scale=1.0)

    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        ##################################################
        # Backbone Forward Pass
        ##################################################
        x = self.backbone.extract_features(data.get('img'))
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)

        ##################################################
        # Hyper-networks Forward Pass
        ##################################################
        t_input = self.hyper_in_t_proj(x)
        hyper_in_h0 = self._swish(self.hyper_in_t_fc_0(t_input))
        hyper_w_t_fc_h0 = self.hypernet_t_fc_h0(hyper_in_h0)
        hyper_in_h1 = self._swish(self.hyper_in_t_fc_1(t_input))
        hyper_w_t_fc_h1 = self.hypernet_t_fc_h1(hyper_in_h1)
        hyper_in_h2 = self._swish(self.hyper_in_t_fc_2(t_input))
        hyper_w_t_fc_h2 = self.hypernet_t_fc_h2(hyper_in_h2)

        rot_input = self.hyper_in_rot_proj(x)
        hyper_in_h0 = self._swish(self.hyper_in_rot_fc_0(rot_input))
        hyper_w_rot_fc_h0 = self.hypernet_rot_fc_h0(hyper_in_h0)
        hyper_in_h1 = self._swish(self.hyper_in_rot_fc_1(rot_input))
        hyper_w_rot_fc_h1 = self.hypernet_rot_fc_h1(hyper_in_h1)
        hyper_in_h2 = self._swish(self.hyper_in_rot_fc_2(rot_input))
        hyper_w_rot_fc_h2 = self.hypernet_rot_fc_h2(hyper_in_h2)

        self.w_t = {'w_h1': hyper_w_t_fc_h0, 'w_h2': hyper_w_t_fc_h1, 'w_o': hyper_w_t_fc_h2}
        self.w_rot = {'w_h1': hyper_w_rot_fc_h0, 'w_h2': hyper_w_rot_fc_h1, 'w_o': hyper_w_rot_fc_h2}

        ##################################################
        # Regression Forward Pass
        ##################################################
        # (1) Hyper-network's regressors
        p_x_hyper = self.regressor_hyper_t(self._global_desc_t, self.w_t)
        p_q_hyper = self.regressor_hyper_rot(self._global_desc_rot, self.w_rot)

        # (2) Trained regressors
        x = self.dropout(F.relu(self.fc1(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)

        ##################################################
        # Output
        ##################################################
        x_t = torch.add(p_x, p_x_hyper)
        x_rot = torch.add(p_q, p_q_hyper)

        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return expected_pose

