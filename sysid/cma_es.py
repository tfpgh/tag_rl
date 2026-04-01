from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CMAESState:
    mean: np.ndarray
    sigma: float
    cov: np.ndarray
    pc: np.ndarray
    ps: np.ndarray
    generation: int


class CMAES:
    def __init__(
        self, dim: int, population_size: int, seed: int, sigma: float = 1.0
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        self.dim = dim
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        weights = np.log(population_size / 2 + 0.5) - np.log(
            np.arange(1, population_size // 2 + 1)
        )
        self.weights = weights / np.sum(weights)
        self.mu_eff = float(np.sum(self.weights) ** 2 / np.sum(self.weights**2))
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0)
            + self.c_sigma
        )
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0
            * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((dim + 2.0) ** 2 + self.mu_eff),
        )
        self.expected_norm = math.sqrt(dim) * (
            1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)
        )
        self.state = CMAESState(
            mean=np.zeros((dim,), dtype=np.float64),
            sigma=sigma,
            cov=np.eye(dim, dtype=np.float64),
            pc=np.zeros((dim,), dtype=np.float64),
            ps=np.zeros((dim,), dtype=np.float64),
            generation=0,
        )

    def ask(self) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(self.state.cov)
        eigvals = np.maximum(eigvals, 1e-12)
        transform = eigvecs @ np.diag(np.sqrt(eigvals))
        z = self.rng.standard_normal((self.population_size, self.dim))
        self._last_z = z
        self._last_transform = transform
        return self.state.mean + self.state.sigma * (z @ transform.T)

    def tell(self, population: np.ndarray, fitness: np.ndarray) -> None:
        order = np.argsort(fitness)
        selected = population[order[: len(self.weights)]]
        old_mean = self.state.mean
        new_mean = np.sum(selected * self.weights[:, None], axis=0)
        eigvals, eigvecs = np.linalg.eigh(self.state.cov)
        eigvals = np.maximum(eigvals, 1e-12)
        inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        ps = (1.0 - self.c_sigma) * self.state.ps + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (inv_sqrt_cov @ ((new_mean - old_mean) / self.state.sigma))
        norm_ps = np.linalg.norm(ps)
        h_sigma = float(
            norm_ps
            / math.sqrt(
                1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.state.generation + 1))
            )
            < (1.4 + 2.0 / (self.dim + 1.0)) * self.expected_norm
        )
        pc = (1.0 - self.c_c) * self.state.pc + h_sigma * math.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_eff
        ) * ((new_mean - old_mean) / self.state.sigma)

        y = (selected - old_mean) / self.state.sigma
        rank_mu = np.zeros_like(self.state.cov)
        for weight, delta in zip(self.weights, y, strict=True):
            rank_mu += weight * np.outer(delta, delta)

        cov = (
            (1.0 - self.c1 - self.c_mu) * self.state.cov
            + self.c1
            * (
                np.outer(pc, pc)
                + (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c) * self.state.cov
            )
            + self.c_mu * rank_mu
        )
        cov = 0.5 * (cov + cov.T)
        sigma = self.state.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (norm_ps / self.expected_norm - 1.0)
        )
        self.state = CMAESState(
            mean=new_mean,
            sigma=max(sigma, 1e-4),
            cov=cov,
            pc=pc,
            ps=ps,
            generation=self.state.generation + 1,
        )
