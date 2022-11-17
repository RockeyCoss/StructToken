def nchw2nlc2nchw(module, x):
    """Flatten [N, C, H, W] shape tensor `x` to [N, L, C] shape tensor. Use the
    reshaped tensor as the input of `module`, and the convert the output of
    `module`, whose shape is.
    [N, L, C], to [N, C, H, W].
    Args:
        module: (Callable): A callable object the takes a tensor
            with shape [N, L, C] as input.
        x: (Tensor): The input tensor of shape [N, C, H, W].
    Returns:
        Tensor: The output tensor of shape [N, C, H, W].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> norm = nn.LayerNorm(4)
        >>> feature_map = torch.rand(4, 4, 5, 5)
        >>> output = nchw2nlc2nchw(norm, feature_map)
    """
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = module(x)
    x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
    return x


def nlc2nchw2nlc(module, x, hw_shape):
    """Convert [N, L, C] shape tensor `x` to [N, C, H, W] shape tensor. Use the
    reshaped tensor as the input of `module`, and convert the output of
    `module`, whose shape is.
    [N, C, H, W], to [N, L, C].
    Args:
        module: (Callable): A callable object the takes a tensor
            with shape [N, C, H, W] as input.
        x: (Tensor): The input tensor of shape [N, L, C].
        hw_shape: (Sequence[int]): The height and width of the
            feature map with shape [N, C, H, W].
    Returns:
        Tensor: The output tensor of shape [N, L, C].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> conv = nn.Conv2d(16, 16, 3, 1, 1)
        >>> feature_map = torch.rand(4, 25, 16)
        >>> output = nlc2nchw2nlc(conv, feature_map, (5, 5))
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    x = x.transpose(1, 2).reshape(B, C, H, W)
    x = module(x)
    x = x.flatten(2).transpose(1, 2)
    return x