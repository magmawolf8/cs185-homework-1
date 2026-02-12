"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # make a Pytorch sequential
        output_dim = action_dim * chunk_size
        
        layers = list()
        
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        predicted_action_flat = self(state)

        expert_action_flat = action_chunk.view(batch_size, -1)

        return nn.functional.mse_loss(predicted_action_flat, expert_action_flat, reduction="mean")

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        predicted_action_flat = self(state)
        return predicted_action_flat.view(batch_size, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        output_dim = action_dim * chunk_size

        layers = list()

        prev_dim = state_dim + output_dim + 1
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, model_input):
        return self.net(model_input)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        device = state.device
        batch_size = state.shape[0]

        noise = torch.randn_like(action_chunk, device=device)
        tau = torch.rand(batch_size, 1, 1, device=device)

        interpolated_actions = tau * action_chunk + (1 - tau) * noise
        target_velocity = action_chunk - noise

        interpolated_flat = interpolated_actions.view(batch_size, -1)
        tau_flat = tau.view(batch_size, 1)

        model_input = torch.cat([state, interpolated_flat, tau_flat], dim=1)
        predicted_velocity = self(model_input)

        # Reshape back to the right dimensions
        predicted_velocity = predicted_velocity.view(
            batch_size, self.chunk_size, self.action_dim
        )

        return nn.functional.mse_loss(predicted_velocity, target_velocity)


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        device = state.device
        batch_size = state.shape[0]

        actions = torch.randn(
            batch_size, self.chunk_size * self.action_dim, device=device
        )

        step_size = 1.0 / num_steps

        for step in range(num_steps):
            tau = torch.full((batch_size, 1), step * step_size, device=device)

            model_input = torch.cat([state, actions, tau], dim=1)
            velocity = self(model_input)
            
            actions = actions + step_size * velocity

        return actions.view(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
