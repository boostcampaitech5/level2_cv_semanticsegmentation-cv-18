import torch

model = torch.load("", map_location="cpu")
model.eval()
dummy_input = torch.randn(1, 3, 512, 512)

# TorchScript 형식으로 변환
traced_model = torch.jit.trace(model, dummy_input)

# TorchScript 파일로 저장
traced_model.save("model.pt")
