import torch

from habana_frameworks.torch.utils.library_loader import load_habana_module


load_habana_module()
device = torch.device("hpu")

y = torch.tensor([3.2451, 2.2926, 3.4715, 4.7845, 2.1488, 2.1488, 4.8688, 3.6512])
label = torch.tensor([4.0747, 2.8591, 4.2882, 6.1464, 2.5679, 2.4672, 6.1946, 4.6887])

# Results are different with SynapseAI 1.4.0
print(torch.nn.functional.mse_loss(y, label))
print(torch.nn.functional.mse_loss(y.to(device), label.to(device)))
