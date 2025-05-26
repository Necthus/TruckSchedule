from model import ScoreNetwork
import torch
import os
params_path = './result/model_params_0415.pth'
scorenet = ScoreNetwork(1,10)

scorenet.load_state_dict(torch.load(params_path))

