from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Callable

import torch.nn


@dataclass
class WeightDecayGroups:
    weight_decay_params: List[torch.nn.Parameter]
    no_weight_decay_params: List[torch.nn.Parameter]


class ISparselyWeightDecayedModule(torch.nn.Module):

    @abstractmethod
    def get_weight_decay_groups(self) -> WeightDecayGroups:
        pass


class ILanguageModel(torch.nn.Module):

    @abstractmethod
    def get_probs(self, prompt: List[int], n_tokens: int, callback: Callable[[torch.tensor], int]) -> None:
        """
        Generates a sequence of tokens from the given prompt
        :param prompt: the original prompt
        :param n_tokens: the number of tokens to generate
        :param callback: invoked with all logits for each token.
        Must return the chosen token for autoregressive sampling.
        :return: the generated sequence of tokens
        """
        pass
