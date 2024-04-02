import torch
import numpy as np
import gc
import os


class GlobalSettings:
    cache_path = "./cache"
    svd_algo = "torch_cpu"


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


@torch.inference_mode()
def get_lowrank_tuple_torch_cpu(tensor, max_rank):
    t = tensor.float()
    u, s, v = torch.linalg.svd(t)
    u, s, v = u[:, :max_rank], s[:max_rank], v[:max_rank, :]
    l = torch.matmul(u, torch.diag(s))
    del t, u, s
    A, B = (v.t(), l.t())  # tensor.t() ~ AB
    return A, B


@torch.inference_mode()
def get_lowrank_tuple_torch_gpu(tensor, max_rank):
    t = tensor.float().to("cuda")
    u, s, v = torch.linalg.svd(t)
    u, s, v = u.to("cpu"), s.to("cpu"), v.to("cpu")
    u, s, v = u[:, :max_rank], s[:max_rank], v[:max_rank, :]
    l = torch.matmul(u, torch.diag(s))
    del t, u, s
    A, B = (v.t(), l.t())  # tensor.t() ~ AB
    return A, B


def get_lowrank_tuple(tensor, name, max_rank):
    svd_algo = GlobalSettings.svd_algo
    cache_path = GlobalSettings.cache_path

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    svd_path = os.path.join(cache_path, name + "_" + str(max_rank) + ".npy")

    try:
        tmp = np.load(svd_path, allow_pickle=True).item()
        A, B = tmp["A"].float(), tmp["B"].float()
        print("Load {} from {}".format(svd_path, cache_path), flush=True)
    except:
        if svd_algo == "torch_cpu":
            A, B = get_lowrank_tuple_torch_cpu(tensor, max_rank)
        if svd_algo == "torch_gpu":
            A, B = get_lowrank_tuple_torch_gpu(tensor, max_rank)
        print("Start save {} into {}.".format(svd_path, cache_path), flush=True)
        np.save(svd_path, {"A": A.half(), "B": B.half()})
    return A, B


def linear_lowrank(linear_layer, device="cuda", return_layer=True):
    # Low-rank weights
    fp16 = linear_layer.fp16
    max_rank = linear_layer.patch_param["max_rank"]

    weight_cpu = linear_layer.weight.data.cpu()
    A, B = get_lowrank_tuple(weight_cpu, name=linear_layer.name, max_rank=max_rank)
    A, B = (A.half(), B.half()) if (fp16) else (A.float(), B.float())

    linear_layer.A = torch.nn.Parameter(A.contiguous().to(device), requires_grad=True)
    linear_layer.B = torch.nn.Parameter(B.contiguous().to(device), requires_grad=False)

    # Bias
    if linear_layer.bias is not None:
        linear_layer.bias.requires_grad = True
        linear_layer.bias.data = (
            linear_layer.bias.data.half() if (fp16) else linear_layer.bias.data.float()
        )
        linear_layer.bias.data = linear_layer.bias.data.to(device)

    # Forward
    def forward_AB(x):
        out = torch.matmul(torch.matmul(x, linear_layer.A), linear_layer.B)
        if linear_layer.bias is not None:
            out += linear_layer.bias
        return out

    linear_layer.forward = forward_AB

    # Cleanup
    del linear_layer.weight, weight_cpu
    cleanup()

    if return_layer:
        return linear_layer


# Create a low-rank linear layer according to a linear layer
class LowRankLinear(torch.nn.Module):
    def __init__(self, linear_layer, patch_params, device):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.patch_param = patch_params[linear_layer.name.split(".")[2]]
        self.max_rank = self.patch_param["max_rank"]
        self.fp16 = patch_params["common"]["fp16"]
        self.name = "low_rank_linear.{}".format(linear_layer.name)

        if self.max_rank is not None:
            linear_lowrank(self, device=device, return_layer=False)
        else:
            self.forward = lambda x: torch.matmul(x, self.weight.t()) + (0.0 if self.bias is None else self.bias)
            if self.fp16:
                self.weight.data = self.weight.data.half().cuda()
                if self.bias is not None:
                    self.bias.data = self.bias.data.half().cuda()
            else:
                self.weight.data = self.weight.data.float().cuda()
                if self.bias is not None:
                    self.bias.data = self.bias.data.float().cuda()
