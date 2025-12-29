import shutil
import os
import torch
import os.path as osp
from ._registry import get_pretrained_weights_path_for_hf
from ._hf_utils import is_huggingface_connected, set_huggingface_endpoint_status, parse_hf_url


def load_nnunet_pretrained_weights(network, fname):
    # TODO: refactor needed
    if (torch.cuda.is_available()):
        saved_model = torch.load(fname, weights_only=False)
    else:
        saved_model = torch.load(fname, weights_only=False, map_location=torch.device('cpu'))
    pretrained_dict = saved_model['state_dict']

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    new_state_dict = {}
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    pretrained_dict = new_state_dict

    # check all conv_blocks to be consistent
    model_dict = network.state_dict()
    ok = True
    error_msg = ""
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                error_msg = f"{key} not in pretraining or shape {model_dict[key].shape} != expected shape {pretrained_dict[key].shape}"
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError(
            "Pretrained weights are not compatible with the current network architecture, error: "
            + error_msg)


def check_and_download_weights_from_hf_url(hf_url):
    hf_cfg = parse_hf_url(hf_url)
    ckpt_local_path = get_pretrained_weights_path_for_hf(repo_id=hf_cfg['repo_id'],
                                                         filename=hf_cfg['filename'])
    if (osp.exists(ckpt_local_path)):
        print(f"cache found, use pretrained weights in {ckpt_local_path}")
        return ckpt_local_path

    cache_dir = osp.dirname(ckpt_local_path)
    os.makedirs(cache_dir, exist_ok=True)

    if (not is_huggingface_connected()):
        set_huggingface_endpoint_status(True)
    from huggingface_hub import hf_hub_download

    # if cannot connect to hf, use chinese mirror instead
    hf_hub_download(repo_id=hf_cfg['repo_id'], filename=hf_cfg['filename'], local_dir=cache_dir)

    shutil.move(osp.join(cache_dir, hf_cfg['filename']), ckpt_local_path)
    if (not osp.exists(ckpt_local_path)):
        raise FileNotFoundError("cannot find ckpt, please re-download the pretrained weights")
    return ckpt_local_path


def load_pretrained_weights(model, checkpoint_path, mode="nnunet", state_dict_key=None):
    # parse checkpoint_path
    ckpt_local_path = checkpoint_path
    if (checkpoint_path.startswith("https://huggingface.co")):
        ckpt_local_path = check_and_download_weights_from_hf_url(checkpoint_path)
    if (mode == 'nnunet'):
        load_nnunet_pretrained_weights(model, ckpt_local_path)
    elif (mode == 'torch'):
        with open(ckpt_local_path, "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        if (state_dict_key):
            state_dict = state_dict[state_dict_key]
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError(f"mode {mode} for weight loading is not implemented")