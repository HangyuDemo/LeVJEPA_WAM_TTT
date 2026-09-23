"""Fail before propagating invalid loss/state or applying invalid gradients."""

import torch


def ensure_finite(value, label):
    leaves = []

    def collect(node, path):
        if isinstance(node, torch.Tensor):
            if node.is_floating_point() or node.is_complex():
                leaves.append((path, node))
        elif isinstance(node, dict):
            for key, child in node.items():
                collect(child, f"{path}.{key}")
        elif isinstance(node, (tuple, list)):
            for index, child in enumerate(node):
                collect(child, f"{path}[{index}]")

    collect(value, label)
    # One host synchronization per device, not one per layer/parameter.
    devices = {tensor.device for _, tensor in leaves}
    for device in devices:
        group = [(path, tensor) for path, tensor in leaves if tensor.device == device]
        flags = torch.stack([torch.isfinite(tensor.detach()).all() for _, tensor in group])
        if not bool(flags.all()):
            bad = [group[index][0] for index in (~flags).nonzero().flatten().tolist()]
            raise FloatingPointError(f"NaN/Inf detected in {', '.join(bad[:8])}")
