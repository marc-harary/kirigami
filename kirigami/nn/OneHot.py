import torch

class OneHot(nn.Module):
    def forward(self, seq):
        encodings = torch.stack([BASE_DICT[char] for char in seq.lower()])
        return torch.unsqueeze(encodings, 2)
