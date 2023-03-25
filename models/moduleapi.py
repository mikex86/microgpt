from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import torch.nn


@dataclass
class WeightDecayGroups:
    weight_decay_params: List[torch.nn.Parameter]
    no_weight_decay_params: List[torch.nn.Parameter]


class ISparselyWeightDecayedModule(torch.nn.Module):

    @abstractmethod
    def get_weight_decay_groups(self) -> WeightDecayGroups:
        pass
