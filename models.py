#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F

class SwitchGate(nn.Module):

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor):

        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        return gate_scores

class SwitchMoE(nn.Module):

    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, self.dim * self.mult, bias = True),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.dim * self.mult, self.dim, bias = True))
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: torch.Tensor):

        # (batch_size, seq_len, num_experts)
        gate_scores = self.gate(
            x
        )

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output

class MLP(nn.Module):
  def __init__(self,):
    super(MLP, self).__init__()
    self.model = nn.Sequential(
      nn.BatchNorm1d(81 * 3),
      nn.Linear(81 * 3, 8),
      nn.Dropout(),
      nn.GELU(),
      nn.BatchNorm1d(8),
      SwitchMoE(8, 3),
      nn.Dropout(),
      nn.GELU(),
      nn.BatchNorm1d(8),
      nn.Linear(8, 4),
      nn.Dropout(),
      nn.Dropout(),
      nn.BatchNorm1d(4),
      nn.Linear(4,1)
    )
  def forward(self, x):
    return self.model(x)

if __name__ == "__main__":
  x = torch.randn(4,81*3)
  mlp = MLP()
  y = mlp(x)
  print(y.shape)
