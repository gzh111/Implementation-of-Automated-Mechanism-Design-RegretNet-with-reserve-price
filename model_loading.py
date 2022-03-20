import torch
from regretnet.datasets import generate_dataset_1x2
from regretnet.regretnet import RegretNet, train_loop, test_loop

model = RegretNet(1, 2, activation='relu', hidden_layer_size=128,
                      p_activation="full_linear", a_activation="full_linear")

model.load_state_dict(torch.load("model/baseline_distill_linear_1x2.pt"))

