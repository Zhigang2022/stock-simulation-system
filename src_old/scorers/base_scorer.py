from abc import ABC, abstractmethod
import numpy as np

class MomentumScorer(ABC):
    @abstractmethod
    def compute_score(self, prices: np.ndarray) -> tuple[float, dict[str, float]]:
        """
        Compute the strategy score and return a flexible dictionary of diagnostic metrics.

        Returns
        -------
        score : float
            The final scalar value used for cross-sectional ranking.
        metrics : dict[str, float]
            A dictionary containing any arbitrary calculation parameters for auditing.
        """
        pass