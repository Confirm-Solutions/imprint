from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field

import jax.numpy as jnp


@dataclass
class Model(ABC):
    """
    To define a Model, subclass this class and:
    1. Define the `family` attribute as either A) a class constant or B) in the
        __init__ method on a per-instance basis.
    2. If needed, define the `family_params` attribute in the same way as the
        `family` attribute. `family_params` defaults to an empty dictionary.
    3. Define the `sim_batch` method.
    """

    family: str
    family_params: dict = field(default_factory=dict)

    @abstractmethod
    def sim_batch(
        self,
        begin_sim: int,
        end_sim: int,
        theta: jnp.ndarray,
        null_truth: jnp.ndarray,
        detailed: bool = False,
    ):
        """
        This is the main method for defining a simulation.

        Args:
            begin_sim: The first simulation index.
            end_sim: Past the last simulation index. For example, we might
            index a pre-generated `samples` array like
                `samples[begin_sim:end_sim]`.
            theta: The parameter values for which to simulate. Shape is
                (n_tiles, d).
            null_truth: The truthiness for each null hypotheses for each tile. Shape
                is (n_tiles, n_nulls).
            detailed: Whether to return extra information for each simulation.
                This is currently unused but will be used in the future to provide
                visibility inside a simulation. Defaults to False.

        Returns:
            stats: The test statistic for each simulation.
                Shape is (n_tiles, end_sim - begin_sim).
        """
        pass
