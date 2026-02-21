from logger import logger
import torch
import torch.nn as nn

class CreateAndAttachHook:
    def __init__(self,model,layer_stats = []):
        self.model = model
        self.layer_stats = layer_stats
    
    def estimate_flops(self, module, input, output):
        try:
            module_type = type(module).__name__
            if isinstance(module, nn.Linear):
                batch_and_seq = input[0].numel() // input[0].shape[-1]
                return 2 * module.in_features * module.out_features * batch_and_seq
            elif module_type == "Conv1D":
                batch_and_seq = input[0].numel() // input[0].shape[-1]
                # Conv1D stores weight as (in, out) — opposite of nn.Linear
                in_features = module.weight.shape[0]
                out_features = module.weight.shape[1]
                return 2 * in_features * out_features * batch_and_seq
            elif isinstance(module, nn.LayerNorm):
                return 2 * output.numel()
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                return output.numel()
            elif isinstance(module, nn.Embedding):
                return input[0].numel()
        except Exception as e:
            logger.error(f"Flops estimation failed: {str(e)}", context="CreateAndAttachHook.estimate_flops")
        return 0
    
    def create_hook(self,layer_name: str) -> callable:
        def hook(mod,input, output):
            try:
                in_shape = list(input[0].shape) if isinstance(input, torch.Tensor) else "N/A"
            except Exception as e:
                logger.error(f"Input shape extraction failed: {str(e)}", context="CreateAndAttachHook.hook")
                in_shape = "N/A"
            try:
                if isinstance(output, torch.Tensor):
                    out_shape = list(output.shape)
                elif isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
                    out_shape = list(output[0].shape)
                else:
                    out_shape = "N/A"
            except Exception as e:
                logger.error(f"Output shape extraction failed: {str(e)}", context="CreateAndAttachHook.hook")
                out_shape = "N/A"
            
            # logger.info(f"Layer: {layer_name} | Input Shape: {in_shape} | Output Shape: {out_shape}", context="CreateAndAttachHook.hook")
            try:
                flops = self.estimate_flops(mod, input, output)
                params = sum(p.numel() for p in mod.parameters())
            except Exception as e:
                logger.error(f"Flops estimation failed: {str(e)}", context="CreateAndAttachHook.hook")
                flops = 0
                params = 0
            self.layer_stats.append({
                "layer": layer_name,
                "type": type(mod).__name__,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "flops": flops,
                "params": params,
            })
        return hook
    
    def attach_hooks(self):
        hooks = []
        for name, mod in self.model.named_modules():
            if name == "":
                continue 
            try:
                hook_fn = self.create_hook(name)
                hooks.append(mod.register_forward_hook(hook_fn))
                # logger.info(f"Attaching hook to layer: {name} ({type(mod).__name__})", context="CreateAndAttachHook.attach_hooks")
            except Exception as e:
                logger.error(f"Failed to attach hook to layer: {name} ({type(mod).__name__}): {str(e)}", context="CreateAndAttachHook.attach_hooks")
                
        return hooks, self.layer_stats
            
            