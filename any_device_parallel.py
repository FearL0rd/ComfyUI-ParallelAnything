import torch
import torch.nn as nn
import types
import copy
import gc
import weakref
from types import SimpleNamespace
from dataclasses import dataclass, fields, is_dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import comfy.model_management

def check_sm80_support(device_name):
    """Check if CUDA device supports SM_80 (Ampere) or higher."""
    if not device_name.startswith("cuda"):
        return True
    try:
        idx = int(device_name.split(":")[-1])
        capability = torch.cuda.get_device_capability(idx)
        major, _ = capability
        return major >= 8
    except Exception as e:
        print(f"[ParallelAnything] Warning: Could not check compute capability for {device_name}: {e}")
        return False

def disable_flash_xformers(model):
    """Aggressively disable Flash Attention and xFormers on the model."""
    disable_configs = [
        ('set_use_memory_efficient_attention_xformers', False),
        ('set_use_flash_attention_2', False),
        ('disable_xformers_memory_efficient_attention', None),
        ('use_xformers', False),
        ('use_flash_attention', False),
        ('use_flash_attention_2', False),
        ('_use_memory_efficient_attention', False),
        ('_flash_attention_enabled', False),
    ]
    
    for attr_or_method, value in disable_configs:
        if hasattr(model, attr_or_method):
            try:
                method = getattr(model, attr_or_method)
                if callable(method) and value is None:
                    method()
                elif callable(method):
                    method(value)
                else:
                    setattr(model, attr_or_method, value)
            except Exception:
                pass

    # Recursively check modules
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['attn', 'attention', 'transformer']):
            for attr in ['use_xformers', 'use_flash_attention', 'use_flash_attention_2', 
                        '_use_memory_efficient_attention', 'enable_flash', 'enable_xformers']:
                if hasattr(module, attr):
                    try:
                        setattr(module, attr, False)
                    except AttributeError:
                        pass
            
            # Reset attention processor if diffusers-style
            if hasattr(module, 'set_processor'):
                try:
                    from diffusers.models.attention_processor import Attention
                    module.set_processor(Attention())
                except Exception:
                    pass

def clear_flux_caches(model):
    """Clear FLUX-specific cached tensors that depend on batch size or input dimensions."""
    cache_attrs = [
        'img_ids', 'txt_ids', '_img_ids', '_txt_ids', 'cached_img_ids', 'cached_txt_ids',
        'pos_emb', '_pos_emb', 'pos_embed', '_pos_embed', 'cached_pos_emb', 'rope', '_rope',
        'freqs_cis', '_freqs_cis', 'freqs', '_freqs', 'cache', '_cache', 'kv_cache', '_kv_cache',
        'attn_bias', '_attn_bias', 'rope_cache', '_rope_cache', 'freqs_cis_cache', '_freqs_cis_cache',
        'temporal_ids', 'frame_ids', 'video_ids', 'temp_pos_emb',
    ]
    
    cleared_count = 0
    
    def clear_attrs(obj, name_prefix=""):
        nonlocal cleared_count
        for attr in cache_attrs:
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    if val is not None:
                        setattr(obj, attr, None)
                        cleared_count += 1
                except (AttributeError, TypeError):
                    pass
    
    # Clear on model itself
    clear_attrs(model)
    
    # Clear on all submodules
    for name, module in model.named_modules():
        clear_attrs(module, name)
    
    if cleared_count > 0:
        print(f"[ParallelAnything] Cleared {cleared_count} cached tensors")

def cleanup_parallel_model(model_ref):
    """Cleanup function to remove parallel replicas and restore original model."""
    model = model_ref() if isinstance(model_ref, weakref.ref) else model_ref
    if model is None:
        return
    
    if not getattr(model, '_true_parallel_active', False):
        return
    
    print("[ParallelAnything] Cleaning up parallel model...")
    
    # Restore original forward
    if hasattr(model, '_original_forward'):
        try:
            model.forward = model._original_forward
            delattr(model, '_original_forward')
        except Exception:
            pass
    
    # Cleanup replicas
    if hasattr(model, '_parallel_replicas'):
        replicas = model._parallel_replicas
        for dev_name, replica in list(replicas.items()):
            try:
                # Move to CPU to free GPU memory
                if hasattr(replica, 'cpu'):
                    replica.cpu()
                
                # Clear caches
                clear_flux_caches(replica)
                
                # Disable gradient checkpointing
                if hasattr(replica, 'gradient_checkpointing'):
                    replica.gradient_checkpointing = False
                if hasattr(replica, '_gradient_checkpointing_func'):
                    replica._gradient_checkpointing_func = None
                    
            except Exception as e:
                print(f"[ParallelAnything] Warning: Error cleaning up replica on {dev_name}: {e}")
        
        try:
            delattr(model, '_parallel_replicas')
        except Exception:
            pass
    
    # Remove all parallel attributes
    for attr in ['_true_parallel_active', '_parallel_devices', '_parallel_streams', 
                 '_parallel_weights', '_auto_vram_balance']:
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    comfy.model_management.soft_empty_cache()

def extract_model_config(model):
    """Extract initialization config from model instance with FLUX-specific handling."""
    config = {}
    possible_attrs = [
        'in_channels', 'out_channels', 'vec_in_dim', 'context_in_dim', 'hidden_size',
        'mlp_ratio', 'num_heads', 'depth', 'depth_single_blocks', 'depth_single', 'axes_dim',
        'theta', 'patch_size', 'qkv_bias', 'guidance_embed', 'txt_ids_dim', 'img_ids_dim',
        'num_res_blocks', 'attention_resolutions', 'dropout', 'channel_mult', 'num_classes',
        'use_checkpoint', 'num_heads_upsample', 'use_scale_shift_norm', 'resblock_updown',
        'use_new_attention_order', 'adm_in_channels', 'num_noises', 'context_dim', 'n_heads',
        'd_head', 'transformer_depth', 'model_channels', 'max_depth', 'num_frames',
        'temporal_compression', 'temporal_dim', 'video_length',
    ]
    
    for attr in possible_attrs:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if not callable(val) and not isinstance(val, (nn.Module, torch.nn.Parameter, torch.Tensor)):
                    config[attr] = val
            except Exception:
                pass
    
    # Handle params attribute (common in FLUX)
    if hasattr(model, 'params'):
        try:
            params = model.params
            if isinstance(params, dict):
                config.update(params)
            elif is_dataclass(params):
                for field_info in fields(params):
                    try:
                        val = getattr(params, field_info.name)
                        if not isinstance(val, (torch.Tensor, nn.Module)):
                            config[field_info.name] = val
                    except Exception:
                        pass
            else:
                params_dict = vars(params)
                if params_dict:
                    config.update({k: v for k, v in params_dict.items() 
                                 if not isinstance(v, (torch.Tensor, nn.Module))})
        except Exception:
            pass
    
    # Handle config attribute
    if hasattr(model, 'config'):
        try:
            cfg = model.config
            if isinstance(cfg, dict):
                config.update({k: v for k, v in cfg.items() 
                             if not isinstance(v, (torch.Tensor, nn.Module))})
            else:
                cfg_dict = {k: v for k, v in vars(cfg).items() 
                           if not k.startswith('_') and not callable(v) 
                           and not isinstance(v, (torch.Tensor, nn.Module))}
                config.update(cfg_dict)
        except Exception:
            pass
    
    if hasattr(model, 'unet_config') and isinstance(model.unet_config, dict):
        config.update({k: v for k, v in model.unet_config.items() 
                      if not isinstance(v, (torch.Tensor, nn.Module))})
    
    # Filter None values and non-serializable objects
    clean_config = {}
    for k, v in config.items():
        if v is not None and not isinstance(v, (torch.Tensor, nn.Module)):
            try:
                # Test if serializable
                copy.deepcopy(v)
                clean_config[k] = v
            except Exception:
                pass
    
    return clean_config

def clone_dataclass_or_object(obj, target_device=None):
    """Deep copy a dataclass or simple object, handling nested structures."""
    if is_dataclass(obj):
        try:
            field_values = {}
            for field_info in fields(obj):
                try:
                    val = getattr(obj, field_info.name)
                    if isinstance(val, torch.Tensor):
                        field_values[field_info.name] = val.clone().detach()
                    elif is_dataclass(val):
                        field_values[field_info.name] = clone_dataclass_or_object(val)
                    elif isinstance(val, (list, tuple)):
                        new_val = []
                        for v in val:
                            if is_dataclass(v):
                                new_val.append(clone_dataclass_or_object(v))
                            elif isinstance(v, torch.Tensor):
                                new_val.append(v.clone())
                            else:
                                new_val.append(copy.deepcopy(v))
                        field_values[field_info.name] = type(val)(new_val)
                    else:
                        field_values[field_info.name] = copy.deepcopy(val)
                except Exception:
                    field_values[field_info.name] = getattr(obj, field_info.name)
            return obj.__class__(**field_values)
        except Exception:
            pass
    
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj

def safe_getattr(obj, attr, default=None):
    """Safely get attribute, returning default if not exists."""
    return getattr(obj, attr, default)

def clone_module_simple(module, target_device):
    """Simple module cloning that handles Parameters and FLUX-specific attributes."""
    if module is None:
        return None
    
    module_class = module.__class__
    
    # Handle specific layer types explicitly to preserve dtypes and devices
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        try:
            weight = safe_getattr(module, 'weight', None)
            weight_dtype = weight.dtype if weight is not None else torch.float32
            has_bias = safe_getattr(module, 'bias', None) is not None
            
            if isinstance(module, nn.Linear):
                in_features = safe_getattr(module, 'in_features')
                out_features = safe_getattr(module, 'out_features')
                new_mod = nn.Linear(in_features, out_features, bias=has_bias, 
                                  device=target_device, dtype=weight_dtype)
            elif isinstance(module, nn.Conv2d):
                new_mod = nn.Conv2d(
                    safe_getattr(module, 'in_channels'),
                    safe_getattr(module, 'out_channels'),
                    safe_getattr(module, 'kernel_size', 1),
                    stride=safe_getattr(module, 'stride', 1),
                    padding=safe_getattr(module, 'padding', 0),
                    dilation=safe_getattr(module, 'dilation', 1),
                    groups=safe_getattr(module, 'groups', 1),
                    bias=has_bias,
                    padding_mode=safe_getattr(module, 'padding_mode', 'zeros'),
                    device=target_device,
                    dtype=weight_dtype
                )
            elif isinstance(module, nn.Conv1d):
                new_mod = nn.Conv1d(
                    safe_getattr(module, 'in_channels'),
                    safe_getattr(module, 'out_channels'),
                    safe_getattr(module, 'kernel_size', 1),
                    stride=safe_getattr(module, 'stride', 1),
                    padding=safe_getattr(module, 'padding', 0),
                    dilation=safe_getattr(module, 'dilation', 1),
                    groups=safe_getattr(module, 'groups', 1),
                    bias=has_bias,
                    padding_mode=safe_getattr(module, 'padding_mode', 'zeros'),
                    device=target_device,
                    dtype=weight_dtype
                )
            
            with torch.no_grad():
                if weight is not None:
                    new_mod.weight.copy_(weight)
                if has_bias and safe_getattr(module, 'bias') is not None:
                    new_mod.bias.copy_(module.bias)
            return new_mod
        except Exception as e:
            print(f"[ParallelAnything] Warning: Failed to reconstruct {module_class}: {e}")
    
    # Handle normalization layers
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        try:
            weight = safe_getattr(module, 'weight', None)
            weight_dtype = weight.dtype if weight is not None else torch.float32
            
            if isinstance(module, nn.LayerNorm):
                new_mod = nn.LayerNorm(
                    safe_getattr(module, 'normalized_shape'),
                    eps=safe_getattr(module, 'eps', 1e-5),
                    elementwise_affine=safe_getattr(module, 'elementwise_affine', True),
                    device=target_device,
                    dtype=weight_dtype
                )
            elif isinstance(module, nn.BatchNorm2d):
                new_mod = nn.BatchNorm2d(
                    safe_getattr(module, 'num_features'),
                    eps=safe_getattr(module, 'eps', 1e-5),
                    momentum=safe_getattr(module, 'momentum', 0.1),
                    affine=safe_getattr(module, 'affine', True),
                    track_running_stats=safe_getattr(module, 'track_running_stats', True),
                    device=target_device,
                    dtype=weight_dtype
                )
            elif isinstance(module, nn.GroupNorm):
                new_mod = nn.GroupNorm(
                    safe_getattr(module, 'num_groups'),
                    safe_getattr(module, 'num_channels'),
                    eps=safe_getattr(module, 'eps', 1e-5),
                    affine=safe_getattr(module, 'affine', True),
                    device=target_device,
                    dtype=weight_dtype
                )
            
            with torch.no_grad():
                if weight is not None:
                    new_mod.weight.copy_(weight)
                if safe_getattr(module, 'bias') is not None:
                    new_mod.bias.copy_(module.bias)
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    if safe_getattr(module, 'running_mean') is not None:
                        new_mod.running_mean.copy_(module.running_mean)
                    if safe_getattr(module, 'running_var') is not None:
                        new_mod.running_var.copy_(module.running_var)
            return new_mod
        except Exception as e:
            print(f"[ParallelAnything] Warning: Failed to reconstruct norm layer {module_class}: {e}")
    
    # Generic fallback
    try:
        new_mod = module_class.__new__(module_class)
        nn.Module.__init__(new_mod)
        
        # Copy parameters
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                new_param = nn.Parameter(param.clone().detach().to(device=target_device), 
                                       requires_grad=False)
                new_mod.register_parameter(name, new_param)
            else:
                new_mod.register_parameter(name, None)
        
        # Copy buffers
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                new_buffer = buffer.clone().detach().to(device=target_device)
                new_mod.register_buffer(name, new_buffer)
            else:
                new_mod.register_buffer(name, None)
        
        # Recursively clone children
        for name, child in module.named_children():
            if child is not None:
                cloned_child = clone_module_simple(child, target_device)
                new_mod.add_module(name, cloned_child)
        
        # Copy other attributes, excluding PyTorch internal ones
        excluded_attrs = {
            '_parameters', '_buffers', '_modules', '_non_persistent_buffers_set',
            '_backward_pre_hooks', '_backward_hooks', '_forward_pre_hooks', 
            '_forward_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks',
            '_extra_state', '_modules_to_load'
        }
        
        cache_attrs = {
            'img_ids', 'txt_ids', '_img_ids', '_txt_ids', 'cached_img_ids', 
            'cached_txt_ids', 'pos_emb', '_pos_emb', 'pos_embed', '_pos_embed',
            'freqs_cis', '_freqs_cis', 'freqs', '_freqs', 'cache', '_cache',
            'kv_cache', '_kv_cache', 'attn_bias', '_attn_bias'
        }
        
        for key, value in module.__dict__.items():
            if key in excluded_attrs:
                continue
                
            try:
                if is_dataclass(value):
                    setattr(new_mod, key, clone_dataclass_or_object(value))
                elif isinstance(value, torch.Tensor):
                    if key in cache_attrs:
                        setattr(new_mod, key, None)
                    else:
                        setattr(new_mod, key, value.clone().detach().to(target_device))
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    if key in cache_attrs:
                        setattr(new_mod, key, None)
                    else:
                        setattr(new_mod, key, type(value)(t.clone().detach().to(target_device) for t in value))
                else:
                    setattr(new_mod, key, copy.deepcopy(value))
            except Exception:
                pass
        
        return new_mod
    except Exception as e:
        raise RuntimeError(f"Failed to clone module {module_class}: {e}")

def safe_model_clone(source_model, target_device, disable_flash=False):
    """FLUX-safe model cloning with explicit memory cleanup."""
    clear_flux_caches(source_model)
    
    # Determine source device
    try:
        src_device = next(source_model.parameters()).device
    except StopIteration:
        src_device = torch.device('cpu')
    
    # Move to CPU for safe cloning if on GPU
    if src_device.type != 'cpu':
        model_cpu = source_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            comfy.model_management.soft_empty_cache()
    else:
        model_cpu = source_model
    
    model_class = model_cpu.__class__
    replica = None
    method_used = "unknown"
    
    try:
        # Try 1: Deep copy
        try:
            replica = copy.deepcopy(model_cpu)
            replica = replica.to(target_device)
            method_used = "deepcopy"
        except (TypeError, RuntimeError, AttributeError) as e:
            if "pickle" in str(e).lower() or "copy" in str(e).lower():
                raise RuntimeError(f"Deepcopy failed: {e}")
            raise
        
    except RuntimeError:
        # Try 2: Config reconstruction
        try:
            print(f"[ParallelAnything] Deepcopy failed, trying config reconstruction...")
            config = extract_model_config(model_cpu)
            state_dict = model_cpu.state_dict()
            
            if not config:
                raise RuntimeError("Could not extract model config")
            
            try:
                replica = model_class(**config)
            except TypeError as e:
                if "missing" in str(e) and "required" in str(e):
                    try:
                        replica = model_class(config)
                    except Exception:
                        config_obj = SimpleNamespace(**config)
                        replica = model_class(config_obj)
                else:
                    raise
            
            replica.load_state_dict(state_dict, strict=False)
            replica = replica.to(target_device)
            method_used = "config_reconstruction"
            
        except Exception as recon_error:
            print(f"[ParallelAnything] Config reconstruction failed: {recon_error}")
            print(f"[ParallelAnything] Attempting manual recursive cloning...")
            
            # Try 3: Manual recursive cloning
            replica = clone_module_simple(model_cpu, target_device)
            method_used = "manual_recursive_clone"
    
    # Cleanup
    if src_device.type != 'cpu':
        del model_cpu
        gc.collect()
    
    if replica is None:
        raise RuntimeError("All clone methods failed")
    
    # Post-processing
    clear_flux_caches(replica)
    
    # Disable gradient checkpointing to save VRAM
    if hasattr(replica, 'gradient_checkpointing'):
        replica.gradient_checkpointing = False
    if hasattr(replica, '_gradient_checkpointing_func'):
        replica._gradient_checkpointing_func = None
    
    # Handle accelerate hooks/offloading
    if hasattr(replica, 'hooks'):
        replica.hooks = []
    if hasattr(replica, '_hf_hook'):
        replica._hf_hook = None
    
    replica.eval()
    for param in replica.parameters():
        param.requires_grad = False
    
    # Ensure buffers are on correct device
    for buffer in replica.buffers():
        if buffer.device != target_device:
            buffer.data = buffer.data.to(target_device)
    
    if disable_flash:
        disable_flash_xformers(replica)
    
    print(f"[ParallelAnything] Cloned via {method_used} to {target_device}")
    return replica

def get_free_vram(device_name):
    """Get available VRAM in MB for a device."""
    try:
        if device_name.startswith("cuda"):
            idx = int(device_name.split(":")[-1])
            torch.cuda.set_device(idx)
            free_memory = (torch.cuda.get_device_properties(idx).total_memory - 
                          torch.cuda.memory_allocated(idx))
            return free_memory / (1024 ** 2)  # Convert to MB
    except Exception:
        pass
    return 0

def auto_split_batch(batch_size, devices, weights):
    """Adjust split based on available VRAM."""
    if not any(d.startswith("cuda") for d in devices):
        return [max(1, int(batch_size * w)) for w in weights]
    
    vram_avail = []
    for d in devices:
        if d.startswith("cuda"):
            vram_avail.append(get_free_vram(d))
        else:
            vram_avail.append(0)
    
    total_vram = sum(vram_avail)
    if total_vram == 0:
        return [max(1, int(batch_size * w)) for w in weights]
    
    # Mix percentage weights with VRAM availability (70% user pref, 30% VRAM)
    adjusted_weights = []
    for i, (w, vram) in enumerate(zip(weights, vram_avail)):
        if vram > 0:
            vram_weight = vram / total_vram
            adjusted = 0.7 * w + 0.3 * vram_weight
        else:
            adjusted = w
        adjusted_weights.append(adjusted)
    
    # Normalize
    total = sum(adjusted_weights)
    adjusted_weights = [w/total for w in adjusted_weights]
    
    split_sizes = [max(1, int(batch_size * w)) for w in adjusted_weights]
    split_sizes[-1] = batch_size - sum(split_sizes[:-1])  # Adjust last to ensure sum = batch_size
    
    return split_sizes

class ParallelDevice:
    @classmethod
    def get_available_devices(cls):
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                devices.append(f"xpu:{i}")
        try:
            import torch_directml
            for i in range(torch_directml.device_count()):
                devices.append(f"privateuseone:{i}")
        except ImportError:
            pass
        return devices

    @classmethod
    def INPUT_TYPES(s):
        available = s.get_available_devices()
        default = "cuda:0" if any(d.startswith("cuda:0") for d in available) else available[0]
        return {
            "required": {
                "device_id": (available, {
                    "default": default,
                    "tooltip": "Select available compute device (CPU/CUDA/MPS/XPU)"
                }),
                "percentage": ("FLOAT", {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Percentage of batch to process on this device"
                }),
            },
            "optional": {
                "previous_devices": ("DEVICE_CHAIN", {
                    "tooltip": "Connect from another ParallelDevice node to chain multiple GPUs"
                }),
            }
        }

    RETURN_TYPES = ("DEVICE_CHAIN",)
    RETURN_NAMES = ("device_chain",)
    FUNCTION = "add_device"
    CATEGORY = "utils/hardware"
    DESCRIPTION = "Add a GPU/CPU/MPS/XPU device to the parallel processing chain"

    def add_device(self, device_id, percentage, previous_devices=None):
        if previous_devices is None:
            previous_devices = []
        
        config = {
            "device": device_id,
            "percentage": float(percentage),
            "weight": float(percentage) / 100.0
        }
        new_chain = previous_devices.copy()
        new_chain.append(config)
        return (new_chain,)

class ParallelDeviceList:
    @classmethod
    def get_available_devices(cls):
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                devices.append(f"xpu:{i}")
        return devices

    @classmethod
    def INPUT_TYPES(s):
        devices = s.get_available_devices()
        def_dev = "cuda:0" if "cuda:0" in devices else devices[0]
        return {
            "required": {
                "device_1": (devices, {"default": def_dev}),
                "pct_1": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "device_2": (devices, {"default": devices[1] if len(devices) > 1 else def_dev}),
                "pct_2": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "device_3": (devices, {"default": devices[2] if len(devices) > 2 else "cpu"}),
                "pct_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "device_4": (devices, {"default": devices[3] if len(devices) > 3 else "cpu"}),
                "pct_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("DEVICE_CHAIN",)
    RETURN_NAMES = ("device_chain",)
    FUNCTION = "create_list"
    CATEGORY = "utils/hardware"

    def create_list(self, device_1, pct_1, device_2, pct_2, device_3="cpu", pct_3=0, device_4="cpu", pct_4=0):
        chain = []
        devices = [(device_1, pct_1), (device_2, pct_2), (device_3, pct_3), (device_4, pct_4)]
        for dev_str, pct in devices:
            if pct > 0:
                chain.append({
                    "device": dev_str,
                    "percentage": float(pct),
                    "weight": float(pct) / 100.0
                })
        return (chain,)

class ParallelAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "device_chain": ("DEVICE_CHAIN", {"tooltip": "Connect from ParallelDevice nodes"}),
            },
            "optional": {
                "workload_split": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable multi-device processing"
                }),
                "auto_vram_balance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adjust batch split based on available VRAM"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "setup_parallel"
    CATEGORY = "utils/hardware"

    def setup_parallel(self, model, device_chain, workload_split=True, auto_vram_balance=False):
        if model is None or not device_chain:
            return (model,)
        
        # Get the actual diffusion model
        target_model = model.model.diffusion_model
        
        # Cleanup existing parallel setup if present
        if hasattr(target_model, "_true_parallel_active"):
            print("[ParallelAnything] Cleaning up previous parallel setup...")
            cleanup_parallel_model(target_model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                comfy.model_management.soft_empty_cache()
        
        # Validate percentages
        total_pct = sum(item["percentage"] for item in device_chain)
        if total_pct <= 0:
            return (model,)
        
        device_names = []
        weights = []
        for item in device_chain:
            weights.append(item["percentage"] / total_pct)
            device_names.append(item["device"])
        
        print(f"[ParallelAnything] Setup: {list(zip(device_names, [f'{w*100:.1f}%' for w in weights]))}")
        
        # Check for devices needing safe attention (older GPUs)
        devices_needing_safe_attention = {}
        for dev_name in device_names:
            if not check_sm80_support(dev_name):
                devices_needing_safe_attention[dev_name] = True
                print(f"[ParallelAnything] {dev_name} < SM_80, disabling Flash/xFormers")
        
        # Validate devices
        for dev in device_names:
            try:
                torch.device(dev)
            except Exception:
                print(f"[ParallelAnything] Invalid device: {dev}")
                return (model,)
        
        # Store original device for restoration
        try:
            original_device = next(target_model.parameters()).device
        except StopIteration:
            original_device = torch.device('cpu')
        
        replicas = {}
        streams = {}
        
        try:
            print(f"[ParallelAnything] Cloning to {len(device_names)} devices...")
            clear_flux_caches(target_model)
            
            # Move original to CPU to free VRAM for cloning
            if original_device.type == 'cuda':
                print(f"[ParallelAnything] Moving model to CPU for safe cloning...")
                target_model = target_model.cpu()
                torch.cuda.empty_cache()
                comfy.model_management.soft_empty_cache()
            
            # Clone to each device
            for dev_name in device_names:
                dev = torch.device(dev_name)
                need_safe = dev_name in devices_needing_safe_attention
                
                try:
                    replica = safe_model_clone(target_model, dev, disable_flash=need_safe)
                    clear_flux_caches(replica)
                    replicas[dev_name] = replica
                    
                    # Create CUDA stream for this device
                    if dev.type == 'cuda':
                        streams[dev_name] = torch.cuda.Stream(dev)
                    else:
                        streams[dev_name] = None
                    
                    print(f"[ParallelAnything] ✓ {dev_name}" + (" (Safe mode)" if need_safe else ""))
                    
                    # Clear cache between clones to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        comfy.model_management.soft_empty_cache()
                        
                except Exception as e:
                    print(f"[ParallelAnything] Error cloning to {dev_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Restore original model to its device
            try:
                if original_device.type == 'cuda':
                    print(f"[ParallelAnything] Restoring original model to {original_device}")
                    target_model = target_model.to(original_device)
            except Exception as e:
                print(f"[ParallelAnything] Warning: Could not restore original model device: {e}")
                
        except RuntimeError as e:
            print(f"[ParallelAnything] VRAM Error during cloning: {e}")
            # Cleanup partial replicas
            for r in replicas.values():
                try:
                    r.cpu()
                except Exception:
                    pass
            try:
                if original_device.type == 'cuda':
                    target_model = target_model.to(original_device)
            except Exception:
                pass
            return (model,)
        
        # Setup parallel forward function
        # Store references in closure
        replicas_ref = replicas
        devices_ref = device_names
        weights_ref = weights
        lead_device = torch.device(device_names[0])
        streams_ref = streams
        auto_balance_ref = auto_vram_balance
        
        def get_batch_size(x):
            """Handle both tensor and list inputs with validation."""
            if isinstance(x, torch.Tensor):
                return x.shape[0]
            elif isinstance(x, (list, tuple)) and len(x) > 0:
                if isinstance(x[0], torch.Tensor):
                    batch_sizes = [t.shape[0] for t in x if isinstance(t, torch.Tensor)]
                    if batch_sizes:
                        if not all(b == batch_sizes[0] for b in batch_sizes):
                            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
                        return batch_sizes[0]
                return len(x)
            else:
                return 1
        
        def split_batch(x, split_sizes):
            """Split tensors or lists of tensors."""
            if isinstance(x, torch.Tensor):
                return torch.split(x, split_sizes, dim=0)
            elif isinstance(x, (list, tuple)):
                # Split each tensor in the list
                split_lists = []
                for t in x:
                    if isinstance(t, torch.Tensor):
                        split_lists.append(torch.split(t, split_sizes, dim=0))
                    else:
                        # Non-tensor, broadcast
                        split_lists.append([t] * len(split_sizes))
                
                # Transpose: list of splits -> splits of lists
                result = []
                for i in range(len(split_sizes)):
                    result.append(type(x)(sl[i] for sl in split_lists))
                return result
            else:
                # Non-tensor, broadcast to all
                return [x] * len(split_sizes)
        
        def move_to_device(x, device, non_blocking=True):
            """Move tensor or list/tuple of tensors to device."""
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=non_blocking)
            elif isinstance(x, (list, tuple)):
                moved = [move_to_device(t, device, non_blocking) for t in x]
                return type(x)(moved)
            else:
                return x
        
        def split_kwargs(kwargs, split_sizes, total_batch):
            """Split keyword arguments that contain batch tensors."""
            split_kwargs_list = [{} for _ in range(len(split_sizes))]
            
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == total_batch:
                    chunks = torch.split(value, split_sizes, dim=0)
                    for i, chunk in enumerate(chunks):
                        split_kwargs_list[i][key] = chunk
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    if all(isinstance(t, torch.Tensor) and t.shape[0] == total_batch for t in value):
                        split_lists = [torch.split(t, split_sizes, dim=0) for t in value]
                        for i in range(len(split_sizes)):
                            split_kwargs_list[i][key] = type(value)(sl[i] for sl in split_lists)
                    else:
                        # Broadcast whole list
                        for i in range(len(split_sizes)):
                            split_kwargs_list[i][key] = value
                else:
                    # Broadcast non-tensor or non-batch-tensor
                    for i in range(len(split_sizes)):
                        split_kwargs_list[i][key] = value
            
            return split_kwargs_list
        
        def concatenate_results(results, dim=0):
            """Concatenate tensor or list/tuple of tensors."""
            if len(results) == 0:
                return results
            
            first = results[0]
            if isinstance(first, torch.Tensor):
                return torch.cat(results, dim=dim)
            elif isinstance(first, (list, tuple)):
                concatenated = []
                for i in range(len(first)):
                    if isinstance(first[i], torch.Tensor):
                        to_concat = [r[i] for r in results]
                        concatenated.append(torch.cat(to_concat, dim=dim))
                    else:
                        concatenated.append(first[i])
                return type(first)(concatenated)
            else:
                return results
        
        def parallel_forward(self, x, timesteps, context=None, **kwargs):
            # Validate batch size
            try:
                batch_size = get_batch_size(x)
            except ValueError as e:
                print(f"[ParallelAnything] Error: {e}")
                raise
            
            # Fallback to single device if batch too small
            if batch_size < len(devices_ref) or not workload_split:
                with torch.no_grad():
                    return replicas_ref[devices_ref[0]](x, timesteps, context=context, **kwargs)
            
            # Calculate split sizes
            if auto_balance_ref:
                split_sizes = auto_split_batch(batch_size, devices_ref, weights_ref)
            else:
                split_sizes = [max(1, int(batch_size * w)) for w in weights_ref]
                split_sizes[-1] = batch_size - sum(split_sizes[:-1])
            
            # Build active device list
            active = []
            for idx, (dev_name, size) in enumerate(zip(devices_ref, split_sizes)):
                if size > 0:
                    active.append({
                        'idx': idx,
                        'dev_name': dev_name,
                        'device': torch.device(dev_name),
                        'replica': replicas_ref[dev_name],
                        'stream': streams_ref.get(dev_name),
                        'size': size,
                    })
            
            if len(active) == 1:
                with torch.no_grad():
                    return active[0]['replica'](x, timesteps, context=context, **kwargs)
            
            # Split inputs
            x_chunks = split_batch(x, [a['size'] for a in active])
            t_chunks = split_batch(timesteps, [a['size'] for a in active])
            
            if context is not None:
                c_chunks = split_batch(context, [a['size'] for a in active])
            else:
                c_chunks = [None] * len(active)
            
            kwargs_chunks = split_kwargs(kwargs, [a['size'] for a in active], batch_size)
            
            results = [None] * len(active)
            exceptions = []
            
            def worker(task_idx):
                task = active[task_idx]
                dev = task['device']
                replica = task['replica']
                stream = task['stream']
                
                try:
                    # Move inputs to device
                    x_in = move_to_device(x_chunks[task_idx], dev)
                    t_in = move_to_device(t_chunks[task_idx], dev)
                    c_in = move_to_device(c_chunks[task_idx], dev) if c_chunks[task_idx] is not None else None
                    
                    # Move kwargs
                    k_in = {}
                    for k, v in kwargs_chunks[task_idx].items():
                        k_in[k] = move_to_device(v, dev, non_blocking=False)
                    
                    # Execute with appropriate synchronization
                    if dev.type == 'cuda' and stream is not None:
                        with torch.cuda.device(dev):
                            with torch.cuda.stream(stream):
                                torch.cuda.synchronize(dev)
                                with torch.no_grad():
                                    out = replica(x_in, t_in, context=c_in, **k_in)
                                torch.cuda.synchronize(dev)
                    elif dev.type == 'cuda':
                        with torch.cuda.device(dev):
                            torch.cuda.synchronize(dev)
                            with torch.no_grad():
                                out = replica(x_in, t_in, context=c_in, **k_in)
                            torch.cuda.synchronize(dev)
                    elif dev.type == 'xpu':
                        with torch.xpu.device(dev):
                            torch.xpu.synchronize(dev)
                            with torch.no_grad():
                                out = replica(x_in, t_in, context=c_in, **k_in)
                            torch.xpu.synchronize(dev)
                    else:
                        with torch.no_grad():
                            out = replica(x_in, t_in, context=c_in, **k_in)
                    
                    # Move result back to lead device
                    out = move_to_device(out, lead_device, non_blocking=False)
                    return task_idx, out
                    
                except Exception as e:
                    return task_idx, e
            
            # Execute in thread pool
            with ThreadPoolExecutor(max_workers=len(active)) as executor:
                futures = [executor.submit(worker, i) for i in range(len(active))]
                
                for future in as_completed(futures):
                    idx, result = future.result()
                    if isinstance(result, Exception):
                        exceptions.append((active[idx]['dev_name'], result))
                        results[idx] = None
                    else:
                        results[idx] = result
            
            # Handle exceptions
            if exceptions:
                for dev_name, exc in exceptions:
                    print(f"[ParallelAnything] Error on {dev_name}: {exc}")
                raise exceptions[0][1]
            
            # Verify all results present
            if any(r is None for r in results):
                missing = [active[i]['dev_name'] for i, r in enumerate(results) if r is None]
                raise RuntimeError(f"Missing results from devices: {missing}")
            
            return concatenate_results(results, dim=0)
        
        # Replace forward method
        target_model._original_forward = target_model.forward
        target_model.forward = types.MethodType(parallel_forward, target_model)
        target_model._true_parallel_active = True
        target_model._parallel_replicas = replicas
        target_model._parallel_devices = device_names
        target_model._parallel_streams = streams
        target_model._parallel_weights = weights
        target_model._auto_vram_balance = auto_vram_balance
        
        # Register cleanup
        weakref.finalize(model, cleanup_parallel_model, weakref.ref(target_model))
        
        # Set load device to lead device for ComfyUI memory management
        model.load_device = lead_device
        
        return (model,)

NODE_CLASS_MAPPINGS = {
    "ParallelAnything": ParallelAnything,
    "ParallelDevice": ParallelDevice,
    "ParallelDeviceList": ParallelDeviceList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAnything": "Parallel Anything (True Multi-GPU)",
    "ParallelDevice": "Parallel Device Config",
    "ParallelDeviceList": "Parallel Device List (1-4x)",
}
