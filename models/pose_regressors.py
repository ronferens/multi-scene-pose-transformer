from .posenet.PoseNet import PoseNet
from .transposenet.EMSTransPoseNet import EMSTransPoseNet
from  .transposenet.MSTransPoseNet import MSTransPoseNet
from typing import Dict, Tuple


def get_model(model_name: str, backbone_path: str, config: Dict) -> Tuple:
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param backbone_path: (str) path to a .pth backbone
    :param config: (dict) config file
    :return: instance of the model (nn.Module) and the name of the model's folder
    """
    if model_name == 'posenet':
        return PoseNet(backbone_path), 'posenet'
    elif model_name == 'ms-transposenet':
        return MSTransPoseNet(config, backbone_path), 'transposenet'
    elif model_name == 'ems-transposenet':
        return EMSTransPoseNet(config, backbone_path), 'transposenet'
    else:
        raise "{} not supported".format(model_name)