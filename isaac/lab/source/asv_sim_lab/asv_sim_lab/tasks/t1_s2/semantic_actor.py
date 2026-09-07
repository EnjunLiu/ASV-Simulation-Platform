"""Checkpoint-compatible task-conditioned semantic actor used by PPO."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .observation import E_MAX, KIN_DIM, OWL_DIM, RAW_DIM, TASK_DIM

SEMANTIC_DIM = 16
ENTITY_DIM = KIN_DIM + SEMANTIC_DIM


class SemanticProjector(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OWL_DIM, hidden), nn.ReLU(), nn.Linear(hidden, SEMANTIC_DIM))

    def forward(self, appearance: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(appearance), dim=-1, eps=1.0e-8)


class SemanticCanonization(nn.Module):
    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(state.dtype)
        position = state[..., :2] * m
        values, vectors = torch.linalg.eigh(position.transpose(-2, -1) @ position)
        axis = vectors[..., -1]
        alignment = (axis * position.sum(dim=-2)).sum(dim=-1)
        axis = torch.where((alignment < 0).unsqueeze(-1), -axis, axis)
        theta = torch.atan2(axis[..., 1], axis[..., 0])
        theta = torch.where(values[..., -1] - values[..., -2] < self.eps, 0.0, theta)
        c, s = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
        result = state.clone()
        px, py, vx, vy = (state[..., index] for index in range(4))
        result[..., 0], result[..., 1] = px * c + py * s, -px * s + py * c
        result[..., 2], result[..., 3] = vx * c + vy * s, -vx * s + vy * c
        return result * m


class TaskHyperNetwork(nn.Module):
    def __init__(self, task_dim: int, value_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * value_dim)
        )
        self.value_dim = value_dim

    def forward(self, task: torch.Tensor):
        value = self.net(task)
        return value[..., : self.value_dim], value[..., self.value_dim :]


class PolarControlHead(nn.Module):
    def __init__(self, task_dim: int = TASK_DIM, hidden: int = 32) -> None:
        super().__init__()
        self.state = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.task_film = nn.Sequential(nn.Linear(task_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * hidden))
        self.out = nn.Linear(hidden, 2)

    def forward(self, kinematics: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        position, velocity = kinematics[..., :2], kinematics[..., 2:4]
        distance = torch.linalg.norm(position, dim=-1, keepdim=True).clamp_min(1.0e-6)
        radial = position / distance
        tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)
        radial_velocity = (velocity * radial).sum(dim=-1, keepdim=True)
        tangent_velocity = (velocity * tangent).sum(dim=-1, keepdim=True)
        scaled_range = distance / 10.0
        invariant = torch.cat(
            (scaled_range, scaled_range.square(), radial_velocity / 2.0, tangent_velocity / 2.0), dim=-1
        )
        state = self.state(invariant)
        gamma, beta = self.task_film(task).chunk(2, dim=-1)
        coefficient = torch.tanh(self.out(torch.relu(state * (1.0 + gamma) + beta)))
        return coefficient[..., :1] * radial + coefficient[..., 1:] * tangent


class SemanticAttentionActor(nn.Module):
    """Semantic target selector followed by an SO(2)-equivariant control head."""

    def __init__(self, task_dim=TASK_DIM, width=16, max_action=0.5, task_center=None, task_scale=1.0):
        super().__init__()
        self.projector = SemanticProjector()
        self.canonize = SemanticCanonization()
        self.w_q = nn.Linear(ENTITY_DIM, width, bias=False)
        self.w_k = nn.Linear(ENTITY_DIM, width, bias=False)
        self.w_v = nn.Linear(ENTITY_DIM, width, bias=False)
        self.w_o = nn.Linear(width, 1, bias=False)
        self.hypernet = TaskHyperNetwork(task_dim, width)
        self.w_select = nn.Linear(ENTITY_DIM, width, bias=False)
        self.selector_hypernet = nn.Sequential(nn.Linear(task_dim, 32), nn.ReLU(), nn.Linear(32, width))
        self.selector_null = nn.Sequential(nn.Linear(task_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.polar_control = PolarControlHead(task_dim)
        self.task_dim, self.width, self.max_action = int(task_dim), int(width), float(max_action)
        center = torch.zeros(self.task_dim) if task_center is None else torch.as_tensor(task_center).reshape(self.task_dim)
        self.register_buffer("task_center", center.float())
        self.register_buffer("task_scale", torch.tensor(float(task_scale)))
        self.use_polar_control = True

    def forward(self, raw: torch.Tensor, task: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        conditioned_task = (task - self.task_center) * self.task_scale
        semantic = self.projector(raw[..., KIN_DIM:])
        entities = torch.cat((raw[..., :KIN_DIM], semantic), dim=-1) * mask.unsqueeze(-1)
        canonical = self.canonize(entities, mask)
        selector_key = self.w_select(canonical)
        selector_query = self.selector_hypernet(conditioned_task)
        logits = (selector_key * selector_query.unsqueeze(-2)).sum(dim=-1) / math.sqrt(self.width)
        logits = logits.masked_fill(~mask, -1.0e9)
        logits = torch.cat((logits, self.selector_null(conditioned_task)), dim=-1)
        selector = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)[..., :E_MAX]
        mass = selector.sum(dim=-1, keepdim=True)
        selected = (selector.unsqueeze(-1) * entities[..., :4]).sum(dim=-2) / mass.clamp_min(1.0e-8)
        action = self.max_action * self.polar_control(selected, conditioned_task)
        action = torch.where(mass > 1.0e-8, action, torch.zeros_like(action))
        norm = torch.linalg.norm(action, dim=-1, keepdim=True)
        return action * torch.clamp(self.max_action / (norm + 1.0e-8), max=1.0)


def load_semantic_actor(path, map_location=None) -> SemanticAttentionActor:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if blob.get("architecture") != "semantic16_attention_v5":
        raise ValueError("PPO initialization requires a semantic16_attention_v5 checkpoint")
    model = SemanticAttentionActor(
        task_dim=int(blob["task_dim"]), width=int(blob["width"]), max_action=float(blob["max_action"]),
        task_center=blob.get("task_center"), task_scale=float(blob.get("task_scale", 1.0))
    )
    model.load_state_dict(blob["state_dict"], strict=True)
    return model
