# from https://github.com/frgfm/Holocron
# https://github.com/frgfm/Holocron/blob/main/LICENSE

from torch import Tensor
from typing import Any, List, Optional, Union
import torch.nn as nn
import torch
from torch.nn import functional as F

class _Loss(nn.Module):
    '''
    Base class for all losses.

    Args:
        weight (Optional[Union[float, List[float], Tensor]]): Weight to apply to the loss. If a float is provided, the same weight will be applied to all classes. If a list is provided, the weight will be applied to each class in the list. If a tensor is provided, the weight will be applied to each class according to the tensor index. Defaults to None.
        ignore_index (int, optional): Target class to ignore. Defaults to -100.
        reduction (str, optional): Reduction method to apply. Defaults to "mean".

    Raises:
        NotImplementedError: If the reduction method is not implemented.

    '''
    def __init__(
        self,
        weight: Optional[Union[float, List[float], Tensor]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # Cast class weights if possible
        self.weight: Optional[Tensor]
        if isinstance(weight, (float, int)):
            self.register_buffer("weight", torch.Tensor([weight, 1 - weight]))
        elif isinstance(weight, list):
            self.register_buffer("weight", torch.Tensor(weight))
        elif isinstance(weight, Tensor):
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ["none", "mean", "sum"]:
            raise NotImplementedError("argument reduction received an incorrect input")
        self.reduction = reduction


class WeightedLoss(_Loss):

    # The WeightedLoss class is a custom PyTorch loss function that applies a weight to an existing loss function.
    # It extends the _Loss base class in PyTorch, which means it inherits some methods and properties that are useful
    # for loss functions.
    # The __init__ method initializes the WeightedLoss object by taking an existing PyTorch loss function (loss)
    # as input and a weight (weight). It stores loss and weight as instance variables.
    # The forward method is the main computation of the loss function. It takes a variable number of input tensors
    # as input and applies the loss function to them. The result of the loss function is multiplied by the weight
    # value and returned as the final loss value.

    def __init__(self, loss, weight=1.0):
        '''
        Wrapper class around loss function that applies weighted with fixed factor.
        This class helps to balance multiple losses if they have different scales

        Args:
            loss (nn.Module): loss function
            weight (float): weight of the loss function

        Returns:
            nn.Module: weighted loss function
        '''
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        '''
        Compute the weighted loss.

        Args:
            *input (Tensor): input tensors

        Returns:
            Tensor: weighted loss
        '''
        return self.loss(*input) * self.weight


class JointLoss(_Loss):

    # The JointLoss allows you to combine two individual loss functions into a single loss function, which can be
    # useful for training complex machine learning models.
    # The JointLoss class is a custom PyTorch loss function that combines two individual loss functions
    # (first and second) into a single loss function by adding their weighted sum. It extends the _Loss base
    # class in PyTorch, which means it inherits some methods and properties that are useful for loss functions.
    # The __init__ method initializes the JointLoss object by taking two PyTorch loss functions (first and second)
    # as input and their corresponding weights (first_weight and second_weight). It then creates two WeightedLoss
    # objects, passing first and second as the first argument to each one, and first_weight and second_weight as
    # the second argument, respectively. The WeightedLoss class is another custom PyTorch loss function that allows
    # us to apply a weight to a standard PyTorch loss function.
    # The forward method is the main computation of the loss function. It takes a variable number of input tensors
    # as input and applies the first and second loss functions to them. The results of the two loss functions are
    # added together and returned as the final loss value.

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        '''
        Wrap two loss functions into one. This class computes a weighted sum of two losses.

        Args:
            first (nn.Module): first loss function
            second (nn.Module): second loss function
            first_weight (float): weight of the first loss function
            second_weight (float): weight of the second loss function

        Returns:
            nn.Module: joint loss function
        '''
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        '''
        Compute the joint loss.

        Args:
            *input (Tensor): input tensors

        Returns:
            Tensor: joint loss
        '''
        return self.first(*input) + self.second(*input)


def focal_loss(
    x: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    gamma: float = 2.0,
) -> Tensor:
    """Implements the focal loss from
    `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_
    Args:
        x (torch.Tensor[N, K, ...]): input tensor
        target (torch.Tensor[N, ...]): hard target tensor
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
        gamma (float, optional): gamma parameter of focal loss
    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    pt = logpt.exp()

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, target.data.view(-1)) * logpt

    # Loss
    loss = -1 * (1 - pt) ** gamma * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


class FocalLoss(_Loss):
    r"""Implementation of Focal Loss as described in
    `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_.

    While the weighted cross-entropy is described by:

    .. math::
        CE(p_t) = -\alpha_t log(p_t)

    where :math:`\alpha_t` is the loss weight of class :math:`t`,
    and :math:`p_t` is the predicted probability of class :math:`t`.

    the focal loss introduces a modulating factor

    .. math::
        FL(p_t) = -\alpha_t (1 - p_t)^\gamma log(p_t)

    where :math:`\gamma` is a positive focusing parameter.

    Args:
        gamma (float, optional): exponent parameter of the focal loss
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, gamma: float = 2.0, **kwargs: Any) -> None:
        ''' Initializes the focal loss function.

        Args:
            gamma (float, optional): exponent parameter of the focal loss
            weight (torch.Tensor[K], optional): class weight for loss computation
            ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
            reduction (str, optional): type of reduction to apply to the final loss
        '''
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        ''' Computes the focal loss. '''
        return focal_loss(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"