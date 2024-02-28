import torch
import torch.nn as nn
from convolutional import *
import math
import warnings
warnings.filterwarnings("ignore")


class DeepPolyVerifier(nn.Module):
    def __init__(self, net, true_label, inputs):
        super().__init__()
        self.layers = net
        for param in self.layers.parameters():
            param.requires_grad = False

        self.y_idx = true_label

        self.conv_params = []
        self.act_params = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.lower_constraint = []
        self.upper_constraint = []
        # setup alphas here?
        self.alphas = self.setup_alphas(inputs.shape)

        self.i_relu = 0


    def initialize(self):
        self.lower_bounds = []
        self.upper_bounds = []
        self.lower_constraint = []
        self.upper_constraint = []

        self.i_relu = 0
    
    def setup_alphas(self, input_size):
        self.conv_ids = []
        alphas = nn.ParameterList()
        prev_outn = 0
        prev_conv_input = input_size
        for l in self.layers:
            if isinstance(l, nn.ReLU):
                alphas.append(nn.Parameter(torch.randn(prev_outn, requires_grad=True)))
            elif isinstance(l, nn.LeakyReLU):
                alphas.append(nn.Parameter(torch.randn(prev_outn, requires_grad=True)))
            elif isinstance(l, nn.Linear):
                prev_outn = l.out_features
            elif isinstance(l, nn.Conv2d):
                prev_conv_input = l(torch.rand(prev_conv_input)).shape
                prev_outn = int(np.prod(prev_conv_input))

        return alphas

    def forward(self, inputs, eps, help=False):
        """
        Initialize the network bounds projected onto the eps sized L_inf ball, where the constraints are just the bounds
        Process each layer 1 by 1, depending on its type
        Backsubstitute (with early stopping) at the end
        """
        # creating the datastructure to store layerwise constrains (considering the number of channels)
        # but flattening along the kernel

        self.initialize()
        self.input_size = inputs.shape
        self.lower_bounds.append(torch.clamp(inputs - eps, min=0.0))
        self.upper_bounds.append(torch.clamp(inputs + eps, max=1.0))

        if isinstance(self.layers[0], nn.Conv2d):
            self.lower_constraint.append(torch.cat((self.lower_bounds[0].flatten().unsqueeze(0), torch.zeros(self.lower_bounds[0].flatten().shape).unsqueeze(0)), dim=0).T)
            self.upper_constraint.append(torch.cat((self.upper_bounds[0].flatten().unsqueeze(0), torch.zeros(self.upper_bounds[0].flatten().shape).unsqueeze(0)), dim=0).T)
        else:
            self.lower_constraint.append(self.lower_bounds[0])
            self.upper_constraint.append(self.upper_bounds[0])

        for i,layer in enumerate(self.layers):

            if isinstance(layer, nn.Linear):
                self.process_affine(layer)
                #self.backsubstitute()
            elif isinstance(layer, nn.Flatten):
                after_conv = (i > 0 and (
                    isinstance(self.layers[i-1], nn.Conv2d) or 
                    isinstance(self.layers[i-1], nn.ReLU) or
                    isinstance(self.layers[i-1], nn.LeakyReLU)
                ))
                self.process_flatten(layer, after_conv)
                #self.backsubstitute()
            elif isinstance(layer, nn.Conv2d):
                self.process_conv(layer)
            elif isinstance(layer, nn.ReLU):
                self.alphas[self.i_relu].data.clamp_(min=0.0, max=1.0)
                self.process_relu(layer, self.alphas[self.i_relu])
                self.backsubstitute()
                self.i_relu += 1
            elif isinstance(layer, nn.LeakyReLU):
                if layer.negative_slope < 1:
                    self.alphas[self.i_relu].data.clamp_(min=layer.negative_slope, max=1.0)
                else:
                    self.alphas[self.i_relu].data.clamp_(min=1.0, max=layer.negative_slope)
                self.process_leaky_relu(layer, self.alphas[self.i_relu])
                self.backsubstitute()
                self.i_relu += 1
            else:
                raise NotImplementedError(f'This type of layer {type(layer)} is not implemented yet! Aborting.')

        self.process_affine(self.create_verification_layer(10, self.y_idx))
        self.backsubstitute()
        # final_lb = self.backsubstitute_alt()

        return self.lower_bounds[-1]
        # return final_lb

    def create_verification_layer(self, n_classes: int, true_label: int) -> torch.nn.Linear:
        """
        Generates an output affine layer for the verification task: `y_true - y_other` for each label, enhancing precision.
        Minimizes only the lowerbound of the output.

        Args:
            num_classes (int): The total number of classes.
            true_label (int): The true label for which the verification layer is created.

        Returns:
            torch.nn.Linear: The verification layer.
        """
        output_layer = torch.nn.Linear(n_classes, n_classes - 1)
        torch.nn.init.zeros_(output_layer.bias)
        torch.nn.init.zeros_(output_layer.weight)

        with torch.no_grad():
            output_layer.weight[:, true_label] = 1
            output_layer.weight[:true_label, :true_label].fill_diagonal_(-1)
            output_layer.weight[true_label:, true_label + 1:].fill_diagonal_(-1)

        for param in output_layer.parameters():
            param.requires_grad = False

        return output_layer


    def process_flatten(self, flatten_layer: nn.Flatten, no_effect: bool):
        #  preserve (out_ch,) info: (out_ch, in_ch, w) -> (out_ch, in_ch*w)
        flatten_layer.start_dim=1 
        
        self.lower_bounds[-1] = flatten_layer(self.lower_bounds[-1].unsqueeze(0))
        self.upper_bounds[-1] = flatten_layer(self.upper_bounds[-1].unsqueeze(0))

        if not no_effect:
            #  preserve (out_ch, in_ch) info: (out_ch, in_ch, n, n) -> (out_ch, in_ch, n*n)
            fl = flatten_layer(self.lower_constraint[-1].unsqueeze(0))
            fl = torch.cat((fl, torch.zeros(fl.shape)), dim=0).T
            fu = flatten_layer(self.upper_constraint[-1].unsqueeze(0))
            fu = torch.cat((fu, torch.zeros(fu.shape)), dim=0).T

            self.lower_constraint[-1] = fl
            self.upper_constraint[-1] = fu



    def process_affine(self, linear_layer: nn.Linear):
        """
        Propagate BOX bounds to the next layer by implementing the affine transformer
        bounds:         [l2, u2] + [l1, u1] = [l1 + l2, u1 + u2]
                        [l2, u2] - [l1, u1] = [l2 - u2, u1 - l2]
        constraints:    w * [l, u] + b = [wl+b, wu+b]  

        Store the computed bounds & constraints
        """

        w = linear_layer.weight
        b = linear_layer.bias

        positive_w = torch.where(w >= 0, w, 0)
        negative_w = torch.where(w >= 0, 0, w)

        lb = torch.mm(self.lower_bounds[-1], positive_w.T) + torch.mm(self.upper_bounds[-1], negative_w.T) + b
        ub = torch.mm(self.upper_bounds[-1], positive_w.T) + torch.mm(self.lower_bounds[-1], negative_w.T) + b

        self.lower_bounds.append(lb)
        self.upper_bounds.append(ub)

        # c = (W | b)
        c = torch.cat((w, b.view(-1, 1)), dim=1)
        self.upper_constraint.append(c)
        self.lower_constraint.append(c)

    """
    proof & intuition that it works: https://colab.research.google.com/drive/1458a1BMtxlSwBbgkoxmP6wugAZCnNY7x?usp=sharing
    """
    def process_conv(self, layer: nn.Conv2d):

        # compute the parameters of the trainsformation
        padding = layer.padding[0]
        stride = layer.stride[0]
        input_size = self.input_size
        output_dim = int(np.floor(input_size[-1] + 2*padding - (layer.weight.shape[-1] - 1) - 1) / stride + 1)

        # Transform to linear layer
        trasformed_weights = toeplitz_multichannel(layer.weight, input_size, padding=padding, stride=stride)
        
        # Compute the lower / upperbounds
        LX = F.pad(self.lower_bounds[-1], (padding, padding, padding, padding), "constant", 0)
        UX = F.pad(self.upper_bounds[-1], (padding, padding, padding, padding), "constant", 0) 
        LX = LX.view(input_size[0], -1)
        UX = UX.view(input_size[0], -1)

        lb_out_filters = []
        ub_out_filters = []

        for Wo in trasformed_weights: # iterate through the output channels
            lb_in_channels = []
            ub_in_channels = []
            for i in range(Wo.shape[0]): # iterate through the input channels
                LXi = LX[i, :]
                UXi = UX[i, :]
                Wi = Wo[i, :, :]
                
                positive_w = torch.where(Wi >= 0, Wi, 0)
                negative_w = torch.where(Wi >= 0, 0, Wi)
                
                # matmul as Conv2d for a single channel
                try:
                    lb_in_channels.append(positive_w @ LXi + negative_w @ UXi) 
                    ub_in_channels.append(positive_w @ UXi + negative_w @ LXi) 
                except:
                    print(self.input_size)
                    print(self.lower_bounds[-1].shape)
                    print("pw: ", positive_w.shape)
                    print("nw: ", negative_w.shape)
                    print("LXi: ", LXi.shape)
                    print("UXi: ", UXi.shape)
                    assert()

            FL = torch.stack(lb_in_channels)
            FU = torch.stack(ub_in_channels)

            # sum over input channels
            lb_out_filters.append(torch.sum(FL, dim=0)) 
            ub_out_filters.append(torch.sum(FU, dim=0)) 

        # stack the output channels
        lb = torch.stack(lb_out_filters).reshape(-1, output_dim, output_dim) + layer.bias.reshape(-1, 1, 1)
        ub = torch.stack(ub_out_filters).reshape(-1, output_dim, output_dim) + layer.bias.reshape(-1, 1, 1)
       
        # update the networks input_size - needed for the next conv layer (= ub.shape = layer(torch.randn(input_size)).shape = toeplitz_multiply(..).shape)
        self.input_size = lb.shape

        self.lower_bounds.append(lb)
        self.upper_bounds.append(ub)
        
        # dropping the "padded" parts
        w = trasformed_weights.reshape(*trasformed_weights.shape[:3], input_size[1] + 2, input_size[1] + 2)
        w = w[..., padding:-padding, padding:-padding]
        w = w.reshape(*trasformed_weights.shape[:3], -1)

        b = layer.bias.reshape(*layer.bias.shape, *([1]*len(w.shape[1:])))
        b = b.expand(*w.shape[:-1], 1)

        n_out_ch = w.shape[0]
        n_in_ch = w.shape[1]
        dim_out = w.shape[2]
        dim_in = w.shape[3]
        w_big = torch.zeros((n_out_ch*dim_out, n_in_ch*dim_in + 1))
        for i_out in range(n_out_ch):
            start_out = i_out * dim_out
            end_out = (i_out + 1) * dim_out
            for j_in in range(n_in_ch):
                start_in = j_in * dim_in
                end_in = (j_in + 1) * dim_in
                w_big[start_out:end_out, start_in:end_in] = w[i_out, j_in, :, :]
            w_big[start_out:end_out, -1] = layer.bias[i_out]

        self.upper_constraint.append(w_big)
        self.lower_constraint.append(w_big)


    def backsubstitute(self):
        uc_last = self.upper_constraint[-1]
        lc_last = self.lower_constraint[-1]

        for (uc_prev, lc_prev, ub_prev, lb_prev) in zip(self.upper_constraint[-2::-1], self.lower_constraint[-2::-1], self.upper_bounds[-2::-1], self.lower_bounds[-2::-1]):
            ## upperbound constraints in (W|b) form
            Au = uc_last[:, :-1] # W
            Bu = uc_last[:, -1] # b

            # substituting the upper bound if xi >= 0 else its lowerbound --> since we want to lowerbound the output neuron and show that is still > 0 
            W_u = torch.where(Au.unsqueeze(-1) >= 0, uc_prev, lc_prev)
            S_u = torch.sum(W_u * Au.unsqueeze(-1), dim=1) # susbtituting weight for the previous neurons TODO:probably can be rewritten as a simple matrix multiplication
            S_u[:, -1] += Bu # adding the bias term for the previous neurons

            ## same for the lowerrbound constraints in (W|b) form
            Al = lc_last[:, :-1]
            Bl = lc_last[:, -1]

            S_l = torch.where(Al.unsqueeze(-1) > 0, lc_prev, uc_prev)
            S_l = torch.sum(S_l * Al.unsqueeze(-1), dim=1)
            S_l[:, -1] += Bl

            uc_last = S_u
            lc_last = S_l

            ## early stopping: check if substituting the intermediate lower/upperbounds verifies the constraint
            """E_l = torch.where(Al.unsqueeze(-1) > 0, lb_prev, ub_prev)
            E_l = torch.sum(E_l * Al.unsqueeze(-1), dim=1)
            S_l[:, -1] += Bl

            lb_intermediate = torch.sum(S_l, dim=1)"""


        # after the last substitution for we only have values and no symbolic variables -> sum them up
        uc_last = torch.sum(uc_last, dim=1)
        lc_last = torch.sum(lc_last, dim=1)

        # update the bounds of the neurons in the final layer with the substituted constraints
        self.upper_bounds[-1] = torch.minimum(self.upper_bounds[-1], uc_last)
        self.lower_bounds[-1] = torch.maximum(self.lower_bounds[-1], lc_last)

    def process_relu(self, layer: nn.ReLU, alphas: torch.Tensor):
        """
        Compute the deep poly relaxation for ReLU.
        """
        if len(self.lower_bounds[-1].shape) > 2:
            lb = layer(self.lower_bounds[-1].flatten().unsqueeze(0))
            ub = layer(self.upper_bounds[-1].flatten().unsqueeze(0))
        else:
            lb = layer(self.lower_bounds[-1])
            ub = layer(self.upper_bounds[-1])
        # Lower and upper weights diagonal
        lw_diag = torch.zeros(lb.shape[1])
        uw_diag = torch.zeros(lb.shape[1])
        # Lower and upper biases
        l_bias = torch.zeros(lb.shape[1])
        u_bias = torch.zeros(ub.shape[1])
        # Find indices depending on ReLU crossing
        # ReLU is zero
        # zer_idx = torch.nonzero(torch.le(self.ub, 0), as_tuple=True)
        # Everything is zero, no need to assign
        # lw_diag[zer_idx] = 0.0
        # uw_diag[zer_idx] = 0.0
        # l_bias[zer_idx] = 0.0
        # u_bias[zer_idx] = 0.0

        # ReLU is identity 
        id_idx = torch.nonzero(torch.ge(self.lower_bounds[-1], 0), as_tuple=True)
        lw_diag[id_idx[1]] = 1.0
        uw_diag[id_idx[1]] = 1.0
        # l_bias[zer_idx] = 0.0
        # u_bias[zer_idx] = 0.0

        # ReLU is crossing
        # TODO: Here the lw_diag[cross_idx[1]] values could be optimized. The values should
        # in [0, 1]. If we use slope optimization it's better to not use the area heuristic as well.
        cross_idx = torch.nonzero(torch.logical_and(self.upper_bounds[-1] > 0, self.lower_bounds[-1] < 0), as_tuple=True)
        u_slope = self.upper_bounds[-1] / (self.upper_bounds[-1] - self.lower_bounds[-1])
        lw_diag[cross_idx[1]] = alphas[cross_idx[1]]
        uw_diag[cross_idx[1]] = u_slope[cross_idx]
        # l_bias[cross_idx] = 0.0
        u_bias[cross_idx[1]] = torch.mul(u_slope, -self.lower_bounds[-1])[cross_idx]
        # Add choice of relaxation based on the area of the triangle
        # cross_opt_2_idx = torch.nonzero(torch.logical_and(torch.ge(self.upper_bounds[-1], -self.lower_bounds[-1]), torch.logical_and(self.upper_bounds[-1] > 0, self.lower_bounds[-1] < 0)), as_tuple=True)
        # lw_diag[cross_opt_2_idx[1]] = 1.0

        # ReLU is crossing
        # lw_diag[cross_idx[1]] = 1.0
        # uw_diag[cross_idx[1]] = u_slope[cross_idx]
        # # l_bias[cross_idx] = 0.0
        # u_bias[cross_idx[1]] = torch.mul(u_slope, -self.lb)[cross_idx]

        # Construct upper and lower weight matrices
        LW = torch.diag(lw_diag)
        UW = torch.diag(uw_diag)

        # Append
        lower_con = torch.cat((LW, l_bias.view(-1, 1)), dim=1)
        upper_con = torch.cat((UW, u_bias.view(-1, 1)), dim=1)
        self.lower_bounds.append(lb)
        self.upper_bounds.append(ub)
        self.upper_constraint.append(upper_con)
        self.lower_constraint.append(lower_con)

    def process_leaky_relu(self, layer: nn.LeakyReLU, alphas: torch.Tensor):
        """
        Compute the relaxation for the leaky ReLU.
        For now the lower approximation coincides with the leaky slope.
        TODO: Implement set the lower approximation as a parameter.
        """
        if len(self.lower_bounds[-1].shape) > 2:
            lb = layer(self.lower_bounds[-1].flatten().unsqueeze(0))
            ub = layer(self.upper_bounds[-1].flatten().unsqueeze(0))
        else:
            lb = layer(self.lower_bounds[-1])
            ub = layer(self.upper_bounds[-1])

        # Cases for relu (positive or negative)
        # Lower and upper weights diagonal
        lw_diag = torch.zeros(lb.shape[1])
        uw_diag = torch.zeros(ub.shape[1])
        # Lower and upper biases
        l_bias = torch.zeros(lb.shape[1])
        u_bias = torch.zeros(ub.shape[1])

        # Indices for positive part -- leaky ReLU is identity
        id_idx = torch.nonzero(torch.ge(self.lower_bounds[-1], 0), as_tuple=True)
        lw_diag[id_idx[1]] = 1.0
        uw_diag[id_idx[1]] = 1.0
        # l_bias[id_idx[1]] = 0.0
        # u_bias[id_idx[1]] = 0.0
        
        # Indices for negative -- leaky ReLU is linear
        neg_idx = torch.nonzero(torch.le(self.upper_bounds[-1], 0), as_tuple=True)
        lw_diag[neg_idx[1]] = layer.negative_slope
        uw_diag[neg_idx[1]] = layer.negative_slope

        # Indices for crossing leaky ReLU
        # TODO: If layer.negative_slope < 1, we can optimize for lw_diag[cross_idx[1]]
        # which should be in the interval [layer.negative_slope, 1.0].
        # If layer.negative_slope > 1, we can optimize for lw_diag[cross_idx[1]]
        # which should be in the interval [1.0, layer.negative_slope].
        cross_idx = torch.nonzero(torch.logical_and(self.upper_bounds[-1] > 0, self.lower_bounds[-1] < 0), as_tuple=True)
        u_slope = (self.upper_bounds[-1] - layer.negative_slope * self.lower_bounds[-1]) / (self.upper_bounds[-1] - self.lower_bounds[-1])
        # Choose the upper and lower relational constraints depending on the leaky slope
        if layer.negative_slope < 1:
            uw_diag[cross_idx[1]] = u_slope[cross_idx]
            u_bias[cross_idx[1]] = (self.upper_bounds[-1] - torch.mul(u_slope, self.upper_bounds[-1]))[cross_idx]
            lw_diag[cross_idx[1]] = alphas[cross_idx[1]]
        else:
            lw_diag[cross_idx[1]] = u_slope[cross_idx]
            l_bias[cross_idx[1]] = (self.upper_bounds[-1] - torch.mul(u_slope, self.upper_bounds[-1]))[cross_idx]
            uw_diag[cross_idx[1]] = alphas[cross_idx[1]]

        # Construct upper and lower weight matrices
        LW = torch.diag(lw_diag)
        UW = torch.diag(uw_diag)

        # Append
        upper_con = torch.cat((UW, u_bias.view(-1, 1)), dim=1)
        lower_con = torch.cat((LW, l_bias.view(-1, 1)), dim=1)
        self.lower_bounds.append(lb)
        self.upper_bounds.append(ub)
        self.upper_constraint.append(upper_con)
        self.lower_constraint.append(lower_con)

    """
    Approach for backsubstitution:
    To propagate the affine relational constraints backwards, one needs
    to consider the sign of the entries of the matrix appearing in the constraints.
    For concreteness, consider the relational constraints:
    x_n <= A_n x_{n-1} + a_n
    x_n >= B_n x_{n-1} + b_n
    and assume for simplicity that x_n is scalar.
    For each element [A_n]_i > 0, we take [A_{n-1}]_{i:}
    for each element [A_n]_i > 0, we take [B_{n-1}]_{i:}.
    We stack these (row) vectors together to make a matrix
    multiply each row with [A_n]_i, and then sum (these rows).
    If x_n is a vector simply construct one such matrix for each entry of x_n.
    A similar procedure is followed to generate the bias terms in the relational constraints.
    """
    def backsubstitute_alt(self):
        """
        Perform backsubstitution to improve the bounds on the difference between classes.
        Right now, this simply performs symbolic backsubstitution until the bound
        is verified or the depth is exhausted.
        TODO: After performing one round of backsubstitution, then one has improved upper/lower
        bounds throughout the network. Then, using these improved bounds we should perform
        propagation through the ReLUs/leaky ReLUs once more, since this will result in better
        slopes for the approximation. This operation can be performed multiple times until
        (i) time runs out; (ii) the property is verified; (iii) the bounds stop changing.
        """
        self.LW = self.lower_constraint[-1][:, :-1].clone()
        self.UW = self.upper_constraint[-1][:, :-1].clone()
        self.l_bias = self.lower_constraint[-1][:, -1].clone()
        self.u_bias = self.upper_constraint[-1][:, -1].clone()
        self.lb = None
        self.ub = None

        ii = 0
        certified = False

        while not(certified):
            certified = True

            # Lower weights
            aux_tens_lw = torch.empty((self.LW.shape[1], self.lower_constraint[-2-ii][:, :-1].shape[1], self.LW.shape[0]))
            pos_mask_lw = torch.nonzero(self.LW > 0, as_tuple=True)
            neg_mask_lw = torch.nonzero(torch.le(self.LW,  0), as_tuple=True)
            aux_tens_lw[pos_mask_lw[1], :, pos_mask_lw[0]] = self.lower_constraint[-2-ii][:, :-1][pos_mask_lw[1], :]
            aux_tens_lw[neg_mask_lw[1], :, neg_mask_lw[0]] = self.upper_constraint[-2-ii][:, :-1][neg_mask_lw[1], :]

            # Lower bias
            aux_l_bias = torch.empty_like(self.LW)
            aux_l_bias[pos_mask_lw[0], pos_mask_lw[1]] = self.lower_constraint[-2-ii][:, -1][pos_mask_lw[1]]
            aux_l_bias[neg_mask_lw[0], neg_mask_lw[1]] = self.upper_constraint[-2-ii][:, -1][neg_mask_lw[1]]
            l_bias_new = torch.einsum('ij, ij->i', self.LW, aux_l_bias) + self.l_bias
            self.LW = torch.einsum('oi,ijo->oj',self.LW, aux_tens_lw)

            # Upper weightss
            aux_tens_uw = torch.empty((self.UW.shape[1], self.upper_constraint[-2-ii][:, :-1].shape[1], self.UW.shape[0]))
            pos_mask_uw = torch.nonzero(self.UW > 0, as_tuple=True)
            neg_mask_uw = torch.nonzero(torch.le(self.UW,  0), as_tuple=True)
            aux_tens_uw[pos_mask_uw[1], :, pos_mask_uw[0]] = self.upper_constraint[-2-ii][:, :-1][pos_mask_uw[1], :]
            aux_tens_uw[neg_mask_uw[1], :, neg_mask_uw[0]] = self.lower_constraint[-2-ii][:, :-1][neg_mask_uw[1], :]

            # Upper Bias
            aux_u_bias = torch.empty_like(self.UW)
            aux_u_bias[pos_mask_uw[0], pos_mask_uw[1]] = self.upper_constraint[-2-ii][:, -1][pos_mask_uw[1]]
            aux_u_bias[neg_mask_uw[0], neg_mask_uw[1]] = self.lower_constraint[-2-ii][:, -1][neg_mask_uw[1]]
            u_bias_new = torch.einsum('ij, ij->i', self.UW, aux_u_bias) + self.u_bias
            self.UW = torch.einsum('oi,ijo->oj',self.UW, aux_tens_uw)

            self.l_bias = l_bias_new.clone()
            self.u_bias = u_bias_new.clone()
        
            self.lb = self.lower_bounds[-3-ii].clone()
            self.ub = self.upper_bounds[-3-ii].clone()

            (final_lb, final_ub) = self.propagate_linear_weights()
            if ii == len(self.upper_constraint) - 3:
                return final_lb
            if torch.any(final_lb < 0):
                certified = False
            if not(certified):
                ii += 1
                continue
            return final_lb
    
    def propagate_linear_weights(self):
        # Center bound
        c = 0.5 * (self.ub + self.lb)
        # Width bound
        w = 0.5 * (self.ub - self.lb)
        # Propagate lower bound
        lcenter = torch.matmul(c, self.LW.t())
        lw_out = torch.matmul(w, self.LW.abs().t())

        final_lb = lcenter - lw_out
        if self.l_bias is not None:
            final_lb += self.l_bias

        # Propagate upper bound
        ucenter = torch.matmul(c, self.UW.t())
        uw_out = torch.matmul(w, self.UW.abs().t())

        final_ub = ucenter + uw_out
        if self.u_bias is not None:
            final_ub += self.u_bias
        
        return (final_lb, final_ub)