"""Configuration and serialization utilities."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    class_a: int = 3
    class_b: int = 7
    n_features: int = 4
    sample_limit: int = 200
    test_size: float = 0.25
    maxiter: int = 20
    shots: int = 256
    seed: int = 42

    def validate(self):
        if self.class_a == self.class_b:
            raise ValueError("Choose two different digit classes.")
        if self.n_features not in (2, 4, 8):
            raise ValueError("PCA features must be 2, 4, or 8.")
        if self.sample_limit < 20 or self.maxiter < 1 or self.shots < 1:
            raise ValueError("Sample limit, iterations, and shots must be positive.")
        return self

    def as_dict(self):
        return asdict(self)
