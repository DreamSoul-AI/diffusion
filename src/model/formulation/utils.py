import math
import torch


# def extract(a, t, x_shape):
#     # retreive the data from the buffer according to the timestep and reshape to the shape wanted
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# def get_index_from_list(vals, t, x_shape):
#     """
#     Returns a specific index t of a passed list of values vals
#     while considering the batch dimension.
#     """
#     batch_size = t.shape[0]
#     # Scale t to the appropriate range
#     t_scaled = t * (vals.size(0) - 1)
#     # Convert to integer indices
#     t_int = t_scaled.long()
#     out = vals.gather(-1, t_int)
#     out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
#     return out


def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def expand_to_planes(input, shape, repeat_batch=False):
    """
    Expand input to match the spatial dimensions of shape.
    Optionally repeat across the batch dimension if repeat_batch is True.
    Handles cases where input does not have spatial dimensions.
    """
    # If the batch dimension needs to be repeated to match the target batch size
    if repeat_batch and input.shape[0] == 1:
        # Expand the batch dimension without extra repetitions
        input = input.expand(shape[0], -1)

    # Add spatial dimensions and repeat as necessary to match `shape`
    if input.dim() == 2:  # Assuming input is [batch_size, channels]
        input = input[:, :, None, None]  # Add spatial dimensions: [batch_size, channels, 1, 1]

    # Repeat spatial dimensions to match the target shape (height and width)
    return input.expand(-1, -1, shape[2], shape[3])
