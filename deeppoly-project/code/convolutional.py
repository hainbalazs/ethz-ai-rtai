import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg

# same as: scipy.linalg.toeplitz
def toeplitz_perfilter(row, N, K, stride):
  # N is already padded
  O = int(np.floor(((N - (K - 1) - 1) / stride) + 1))
  # Repeat the kernel matrix J times
  repeated_matrix = row.repeat(O, 1)

  # Pad the matrix to be O*N
  padded_matrix = F.pad(repeated_matrix, (0, N - row.size(0)))

  # Shift them circularry by incrementally increasing amount
  rolled_rows = [torch.roll(padded_matrix, shifts=i*stride, dims=1) for i in range(O)]
  output = torch.stack(rolled_rows)[:, 0, :]

  # Mask out the places of the 0s
  i_indices, j_indices = torch.meshgrid(torch.arange(O), torch.arange(N))
  mask = ((i_indices * stride > j_indices) | (j_indices >= i_indices * stride + row.size()[0]))
  # Apply the mask
  return torch.where(~mask, output, 0)


def toeplitz_perchannel(kernel, input_size, stride=1):
    """ 
    Output dim: (O**2, N**2) 
    where N is already padded
    and O = floor(((N - (K - 1) - 1) / stride) + 1)
    """
    # shapes
    K = kernel.shape[0]
    N = input_size[0]
    O = int(np.floor(((N - (K - 1) - 1) / stride) + 1))

    # from each row of the kernel construct a toeplitz matrix
    W_conv = torch.zeros((O, O, N, N))
    for r in range(K):
        toeplitz = toeplitz_perfilter(kernel[r], N, K, stride)
        #toeplitz2 = torch.tensor(linalg.toeplitz(c=(kernel[r,0], *np.zeros(N-K)), r=(*kernel[r], *np.zeros(N-K))))
        for c in range(O):
          # and create the doubly blocked W
          W_conv[c, :, r+(c*stride), :] = toeplitz

    return W_conv.reshape(O*O, N*N)


def toeplitz_multichannel(kernel, input_size, padding=0, stride=1):
    """Compute toeplitz matrix for 2d conv with multiple in and out channels and batches.
    Input dim: (in_ch, N, N)
    Kernel dim: (out_ch, in_ch, K, K)
    Output dim: (out_ch, in_ch, O**2, N**2)
    where O = floor((N + 2*padding - (K - 1) - 1) / stride + 1)  
    """
    # idea is that for each output channel and input channel we want to create a Toeplitz map (reduce to the single channel case)
    N = input_size[-1]
    K = kernel.shape[-1]
    N_padded = N + 2*padding
    O = int(np.floor(((N + 2*padding - (K - 1) - 1) / stride) + 1))
    output_size = (kernel.shape[0], input_size[0], O**2, N_padded**2)

    T = torch.zeros(output_size)
    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_perchannel(k, (N_padded, N_padded), stride)
            T[i, j, :, :] = T_k

    return T


def toeplitz_multiply(W, B, X, output_dim):
  out_filters = []

  for Wo in W: # iterate through the output channels
    in_channels = []
    for i in range(Wo.shape[0]): # iterate through the input channels
      Xi = X[i, :]
      Wi = Wo[i, :, :]
      in_channels.append(Wi @ Xi) # matmul as Conv2d for a single channel

    F = torch.stack(in_channels)
    out_filters.append(torch.sum(F, dim=0)) # sum over input channels

  O = torch.stack(out_filters).reshape(-1, output_dim, output_dim) # stack the output channels
  B = B.reshape(-1, 1, 1)

  return O + B # add the bias
