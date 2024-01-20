import torch, skimage
from skimage import io, filters
from torchvision import transforms
import numpy as np
import random

def replace_BatchNorm2d(A, B, v=None, replace_bias=True, randomly_select=False, last_vs=None):
    """
    randomly_select (bool): If you have randomly select neurons to replace at the last layer
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    """
    
    if v is None: v = B.num_features
    # print('Replacing BatchNorm2d, v = {}'.format(v))
    
    if last_vs is not None: assert len(last_vs) == v
    else: last_vs = list(range(v))
    # Replace
    A.weight.data[last_vs] = B.weight.data[:v]
    if replace_bias: A.bias.data[last_vs] = B.bias.data[:v]
    A.running_mean.data[last_vs] = B.running_mean.data[:v]
    A.running_var.data[last_vs] = B.running_var.data[:v]
    # print('Replacing BatchNorm2d, A.shape = {}, B.shape = {}, vs = last_vs = {}'.format(A.weight.shape, B.weight.shape, last_vs))
    return last_vs

def replace_Conv2d(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer
    vs (list): Force the neurons' indices selected at this layer to be `vs` (useful in residual connection)
    """
    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, v = {}, last_v = {}'.format(A.weight.shape, B.weight.shape, v, last_v))
    
    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, vs = {}, last_vs = {}'.format(A.weight.shape, B.weight.shape, vs, last_vs))
    return vs

def replace_Linear(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    force_vs (list): Force the neurons' indices selected at this layer to be `force_vs`, only available when `randomly_select` is True
                     (useful in residual connection)
    """

    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]

    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    return vs

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print(output.shape)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].sum().float()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import contextlib

class Interp1d(torch.autograd.Function):
    """
    """
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

def apply_Gotham(inputs):
    device = inputs.device
    sharpen = transforms.RandomAdjustSharpness(sharpness_factor=2)

    def channel_adjust(channel, values):
        orig_size = channel.shape
        flat_channel = channel.flatten()
        adjusted = Interp1d()(torch.linspace(0, 1, len(values)).to(device=channel.device), torch.tensor(values).to(device=channel.device), flat_channel)
        return adjusted.reshape(orig_size)

    r = inputs[:, 0, :, :]
    b = inputs[:, 2, :, :]
    r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    b_more = torch.clip(b -3, 0, 1.0) # 0.03 -> 0.1
    merged = torch.cat((r_boost_lower.unsqueeze(1), inputs[:, 1, :, :].unsqueeze(1), b_more.unsqueeze(1)), dim=1).to(device=device)
    final = sharpen(merged)
    b = final[:, 2, :, :]
    b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
    final[:, 2, :, :] = b_adjusted
    return final.float()

def apply_BlackWhite(inputs):
    """
    `inputs`: tensor of size [batch_size, #channel, width, height]
    """
    device = inputs.device
    inputs = inputs.cpu()

    r = inputs[:, 0, :, :]
    g = inputs[:, 1, :, :]
    b = inputs[:, 2, :, :]
    final = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1).repeat(1, 3, 1, 1).to(device=device)
    return final.float()