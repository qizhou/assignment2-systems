import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print(x.dtype)
        x = self.relu(x)
        print(x.dtype)
        x = self.ln(x)
        print(x.dtype)
        x = self.fc2(x)
        print(x.dtype)
        return x

device="cuda"
# dtype = torch.float16
dtype = torch.bfloat16
model = ToyModel(10, 10).cuda()
x = torch.randn((1, 10), device=device)

with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x)
    z = torch.randn((1, 10), device=device)
    for param in model.parameters():
        print(param.data)
    print(y)
    loss = torch.mean(y - z)
    print(loss)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Layer: {name} | Gradient {param.grad}")