import torch
import torch.nn as nn

class WanActionEncoder(nn.Module):
    def __init__(self, action_dim=7, hidden_dim=512, out_dim=5120, num_layers=4):
        """
        Encodes action sequences into features compatible with WanModel's cross attention.
        """
        super().__init__()
        self.action_dim = action_dim
        # action + delta = 2 * action_dim
        input_dim = 2 * action_dim
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, action_seq):
        delta_raw = action_seq[:, 1:, :] - action_seq[:, :-1, :]
        delta_pad = torch.zeros_like(action_seq[:, :1, :])
        delta = torch.cat([delta_pad, delta_raw], dim=1)
        action_combined = torch.cat([action_seq, delta], dim=-1)  # (b, L, 14)
        action_context = self.mlp(action_combined)  # (b, L, out_dim)
        return action_context

    @staticmethod
    def state_dict_converter():
        return WanActionEncoderStateDictConverter()

class WanActionEncoderStateDictConverter:
    def __init__(self):
        pass
    def from_civitai(self, state_dict):
        return state_dict