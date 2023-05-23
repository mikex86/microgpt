from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Callable, Tuple

import torch.nn
from torch.cuda.amp import GradScaler


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
    def back_propagate(self, x: torch.tensor, targets: torch.tensor,
                       loss_scalar: GradScaler = None,
                       hyper_save_memory: bool = False) -> Tuple[float, torch.Tensor]:
        """
        Back-propagates the cross entropy scaled loss between
        the given targets and the model's predictions and returns the un-scaled loss
        :param x: the input sequence
        :param targets: the target sequence
        :param loss_scalar: the GradScaler used to scale the loss
        :param hyper_save_memory: whether to delete unused tensors and free the cuda cache between each chunk.
        Only use when absolutely necessary to avoid OOM errors.
        :return: the un-scaled loss
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def get_eval_loss(self, x: torch.tensor, y: torch.tensor) -> float:
        """
        Evaluates the model on the given data
        :param x: the input sequence
        :param y: the target sequence
        :return: the loss
        """
        pass

    @abstractmethod
    @torch.inference_mode()
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

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        pass


class BasicLanguageModel(ILanguageModel, ABC):

    @torch.inference_mode()
    def get_probs(self, prompt: List[int], n_tokens: int, callback: Callable[[torch.tensor], int]) -> None:
        self.eval()
        device = next(self.parameters()).device
        tokens = prompt.copy()
        for _ in range(n_tokens):
            tokens = tokens[-self.max_seq_length:]  # crop to block size
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = self(tokens_tensor)
            logits = logits[0, -1, :]
            token = callback(logits)
            tokens.append(token)
        self.train()

    def back_propagate(self, x: torch.tensor, targets: torch.tensor, loss_scalar: GradScaler = None,
                       hyper_save_memory: bool = False) -> Tuple[float, torch.Tensor]:
        self.train()
        device = next(self.parameters()).device
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        unscaled_loss = loss.item()

        if loss_scalar is not None:
            loss = loss_scalar.scale(loss)
        loss.backward()

        logits_copy = logits.detach().clone()

        if hyper_save_memory:
            del logits
            del loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return unscaled_loss, logits_copy

    @torch.no_grad()
    def get_eval_loss(self, x: torch.tensor, y: torch.tensor) -> float:
        self.eval()
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss_item = loss.item()

        del logits
        del loss

        return loss_item


    @property
    @abstractmethod
    def max_seq_length(self) -> int:
        pass
