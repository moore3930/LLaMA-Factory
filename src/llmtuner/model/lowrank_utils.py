import os.path
import torch
import bitsandbytes as bnb
from torch.nn import Linear
from .low_rank_linear import LowRankLinear
import tqdm
import numpy as np
import gc


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def patch_linear(linear_layer, name, res_dir, config, device):
    linear_layer.name = name
    weight = linear_layer.weight.data.to(device)
    low_rank_linear_layer = LowRankLinear(linear_layer, config, device=device)

    # TODO: if config[x][max_rank] is None, no need to do pruning
    delta = weight - torch.matmul(low_rank_linear_layer.A, low_rank_linear_layer.B).t()
    torch.save(delta, os.path.join(res_dir, "{}.{}.pt".format(low_rank_linear_layer.name, "delta")))
    cleanup()
    return low_rank_linear_layer


def recover_linear(linear_layer, res_dir):
    # TODO: if config[x][max_rank] is None, no need to do recovering
    delta = torch.load(os.path.join(res_dir, "{}.{}.pt".format(linear_layer.name, "delta")))
    weight = (torch.matmul(linear_layer.A, linear_layer.B).t() + delta).clone()
    if linear_layer.bias is not None:
        bias = linear_layer.bias.clone()
        recovered_linear_layer = Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=True)
        recovered_linear_layer.weight.data = weight
        recovered_linear_layer.bias.data = bias
    else:
        recovered_linear_layer = Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=False)
        recovered_linear_layer.weight.data = weight
    cleanup()
    return recovered_linear_layer


def low_rank_pruning(model, res_dir, config, is_train, device):
    layers = model.model.layers

    # If training, convert to low-rank version. Otherwise, convert back to dense model for inference.
    if is_train:
        for i in range(len(layers)):
            linear_name = "{}.{}.{}".format(i, config["up_proj"]["max_rank"], "up_proj")
            layers[i].mlp.up_proj = patch_linear(layers[i].mlp.up_proj, linear_name, res_dir, config, device)

            linear_name = "{}.{}.{}".format(i, config["down_proj"]["max_rank"], "down_proj")
            layers[i].mlp.down_proj = patch_linear(layers[i].mlp.down_proj, linear_name, res_dir, config, device)

            linear_name = "{}.{}.{}".format(i, config["gate_proj"]["max_rank"], "gate_proj")
            layers[i].mlp.gate_proj = patch_linear(layers[i].mlp.gate_proj, linear_name, res_dir, config, device)

            linear_name = "{}.{}.{}".format(i, config["q_proj"]["max_rank"], "q_proj")
            layers[i].self_attn.q_proj = patch_linear(layers[i].self_attn.q_proj, linear_name, res_dir, config, device)

            linear_name = "{}.{}.{}".format(i, config["k_proj"]["max_rank"], "k_proj")
            layers[i].self_attn.k_proj = patch_linear(layers[i].self_attn.k_proj, linear_name, res_dir, config, device)

            linear_name = "{}.{}.{}".format(i, config["v_proj"]["max_rank"], "v_proj")
            layers[i].self_attn.v_proj = patch_linear(layers[i].self_attn.v_proj, linear_name, res_dir, config, device)

            linear_name = "{}.{}.{}".format(i, config["o_proj"]["max_rank"], "o_proj")
            layers[i].self_attn.o_proj = patch_linear(layers[i].self_attn.o_proj, linear_name, res_dir, config, device)

    else:
        for i in range(len(layers)):
            recovered_linear_layer = recover_linear(layers[i].mlp.up_proj, res_dir)
            layers[i].mlp.up_proj = recovered_linear_layer

            recovered_linear_layer = recover_linear(layers[i].mlp.down_proj, res_dir)
            layers[i].mlp.down_proj = recovered_linear_layer

            recovered_linear_layer = recover_linear(layers[i].mlp.gate_proj, res_dir)
            layers[i].mlp.gate_proj = recovered_linear_layer

            recovered_linear_layer = recover_linear(layers[i].self_attn.q_proj, res_dir)
            layers[i].self_attn.q_proj = recovered_linear_layer

            recovered_linear_layer = recover_linear(layers[i].self_attn.k_proj, res_dir)
            layers[i].self_attn.k_proj = recovered_linear_layer

            recovered_linear_layer = recover_linear(layers[i].self_attn.v_proj, res_dir)
            layers[i].self_attn.v_proj = recovered_linear_layer

            recovered_linear_layer = recover_linear(layers[i].self_attn.o_proj, res_dir)
            layers[i].self_attn.o_proj = recovered_linear_layer

    return model


def low_rank_pruning_old(model, res_dir, config, is_train, device):
    layers = model.model.layers

    # If training, convert to low-rank version. Otherwise, convert back to dense model for inference.
    if is_train:
        for i in range(len(layers)):
            weight = layers[i].mlp.up_proj.weight.data.to(device)
            layers[i].mlp.up_proj.name = "{}.{}".format(i, "up_proj")
            layers[i].mlp.up_proj = LowRankLinear(layers[i].mlp.up_proj, config, device=device)
            delta = weight - torch.matmul(layers[i].mlp.up_proj.A, layers[i].mlp.up_proj.B).t()
            torch.save(delta, os.path.join(res_dir, "{}.{}.pt".format(layers[i].mlp.up_proj.name, "delta")))

        for i in range(len(layers)):
            weight = layers[i].mlp.down_proj.weight.data.to(device)
            layers[i].mlp.down_proj.name = "{}.{}".format(i, "down_proj")
            layers[i].mlp.down_proj = LowRankLinear(layers[i].mlp.down_proj, config, device=device)
            delta = weight - torch.matmul(layers[i].mlp.down_proj.A, layers[i].mlp.down_proj.B).t()
            torch.save(delta, os.path.join(res_dir, "{}.{}.pt".format(layers[i].mlp.down_proj.name, "delta")))
    else:
        for i in range(len(layers)):
            delta = torch.load(os.path.join(res_dir, "{}.{}.pt".format(layers[i].mlp.up_proj.name, "delta")))
            weight = (torch.matmul(layers[i].mlp.up_proj.A, layers[i].mlp.up_proj.B).t() + delta).clone()
            if layers[i].mlp.up_proj.bias is not None:
                bias = layers[i].mlp.up_proj.bias.clone()
                new_linear = Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=True)
                new_linear.weight.data = weight
                new_linear.bias.data = bias
            else:
                new_linear = Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=False)
                new_linear.weight.data = weight
            layers[i].mlp.up_proj = new_linear

        for i in range(len(layers)):
            delta = torch.load(os.path.join(res_dir, "{}.{}.pt".format(layers[i].mlp.down_proj.name, "delta")))
            weight = (torch.matmul(layers[i].mlp.down_proj.A, layers[i].mlp.down_proj.B).t() + delta).clone()
            if layers[i].mlp.down_proj.bias is not None:
                bias = layers[i].mlp.down_proj.bias.clone()
                new_linear = Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=True)
                new_linear.weight.data = weight
                new_linear.bias.data = bias
            else:
                new_linear = Linear(in_features=weight.shape[0], out_features=weight.shape[1], bias=False)
                new_linear.weight.data = weight
            layers[i].mlp.down_proj = new_linear
    torch.cuda.empty_cache()
    gc.collect()
    return model


def low_rank_pruning_using_state_dict(model, res_dir, config):
    state_dict = model.state_dict()
    print(state_dict.keys())

    # Get all linear layers
    linear_layer_list = []
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            linear_layer_list.append(name)

    # tensor1 = model.model.decoder.project_in.weight
    # state_dict1 = model.state_dict()
    # model.model.decoder.project_in = torch.nn.Linear(in_features=512, out_features=1024)
    # # new_linear = torch.nn.Linear(in_features=512, out_features=1024)
    # # setattr(model, "model.decoder.project_in", new_linear)
    # tensor2 = model.model.decoder.project_in.weight
    # state_dict2 = model.state_dict()
    #
    # if not torch.equal(tensor1, tensor2):
    #     print("NO")
    # else:
    #     print("YES")
    #
    # print(tensor1)
    # print(tensor2)
    #
    # models_match = True
    # for key in state_dict1.keys():
    #     if not torch.equal(state_dict1[key], state_dict2[key]):
    #         models_match = False
    #         print(state_dict1[key])
    #         print(state_dict2[key])
    # print(models_match)

    # Set modules
    for name in linear_layer_list:
        desired_module = model
        module_names = name.split('.')[:-1]
        for module_name in module_names:
            desired_module = getattr(desired_module, module_name, None)

        # Set module
        last_name = name.split('.')[-1]
        if desired_module is not None:
            temp = getattr(desired_module, last_name, None)
            new_linear_module = torch.nn.Linear(in_features=temp.in_features,
                                                out_features=temp.out_features,
                                                bias=False)
            setattr(model, name, new_linear_module)

            # update state_dict, otherwise, the parameter will not change.
            state_dict[name + ".weight"] = new_linear_module.weight
            if name + ".bias" in state_dict.keys():
                state_dict.pop(name + ".bias", None)

    model.load_state_dict(state_dict)

        # module = LowRankLinear(module, name, device='cuda', patch_params=config)

    return model


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            lora_module_names.add(name)
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )


def formatting_prompts_func(example):
  output_texts = []
  for i in range(len(example['prompt'])):
      text = f"An AI tool that corrects and rephrase user text grammar errors delimited by triple backticks to standard English.\n### Input: ```{example['prompt'][i]}```\n ### Output: {example['completion'][i]}"
      output_texts.append(text)
  return output_texts