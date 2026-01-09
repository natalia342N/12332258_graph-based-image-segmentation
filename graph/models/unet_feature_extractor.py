import torch
import torch.nn as nn

class UNetFeatureExtractor(nn.Module):
    def __init__(self, unet: nn.Module, hook_module_name: str):
        super().__init__()
        self.unet = unet
        self.hook_module_name = hook_module_name
        self._feat = None
        name_to_module = dict(self.unet.named_modules())
        assert hook_module_name in name_to_module, (
            f"Module '{hook_module_name}' not found. Available: "
            f"{list(name_to_module.keys())[:30]} ..."
        )
        self.hook_module = name_to_module[hook_module_name]
        self.hook_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._feat = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.unet(x)
        assert self._feat is not None, "Hook did not capture any feature map."
        return self._feat
