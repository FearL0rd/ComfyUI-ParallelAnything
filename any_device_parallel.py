import torch
import torch.nn as nn
import types
import copy
import io
import pickle
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
        major, minor = capability
        return major >= 8
    except Exception as e:
        print(f"[ParallelAnything] Warning: Could not check compute capability for {device_name}: {e}")
        return False

def disable_flash_xformers(model):
    """Aggressively disable Flash Attention and xFormers on the model."""
    disable_methods = [
        ('set_use_memory_efficient_attention_xformers', False),
        ('set_use_flash_attention_2', False),
        ('disable_xformers_memory_efficient_attention', None),
        ('use_xformers', False),
        ('use_flash_attention', False),
        ('use_flash_attention_2', False),
        ('_use_memory_efficient_attention', False),
        ('_flash_attention_enabled', False),
    ]
    for attr_or_method, value in disable_methods:
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
    
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['attn', 'attention', 'transformer']):
            for attr in ['use_xformers', 'use_flash_attention', 'use_flash_attention_2', 
                        '_use_memory_efficient_attention', 'enable_flash', 'enable_xformers']:
                if hasattr(module, attr):
                    try:
                        setattr(module, attr, False)
                    except:
                        pass
            if hasattr(module, 'set_processor'):
                try:
                    from diffusers.models.attention_processor import Attention
                    module.set_processor(Attention())
                except:
                    pass

def clear_flux_caches(model):
    """Clear FLUX-specific cached tensors that depend on batch size or input dimensions."""
    cache_attrs = [
        'img_ids', 'txt_ids', '_img_ids', '_txt_ids', 'cached_img_ids', 'cached_txt_ids',
        'pos_emb', '_pos_emb', 'pos_embed', '_pos_embed', 'cached_pos_emb',
        'rope', '_rope', 'freqs_cis', '_freqs_cis', 'freqs', '_freqs',
        'cache', '_cache', 'kv_cache', '_kv_cache', 'attn_bias', '_attn_bias',
    ]
    cleared = []
    for name, module in model.named_modules():
        for attr in cache_attrs:
            if hasattr(module, attr):
                try:
                    val = getattr(module, attr)
                    if val is not None:
                        setattr(module, attr, None)
                        cleared.append(f"{name}.{attr}" if name else attr)
                except Exception:
                    pass
    for attr in cache_attrs:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if val is not None:
                    setattr(model, attr, None)
                    cleared.append(attr)
            except Exception:
                pass
    if cleared:
        print(f"[ParallelAnything] Cleared {len(cleared)} cached tensors")

def cleanup_parallel_model(model):
    """Cleanup function to remove parallel replicas and restore original model."""
    if not hasattr(model, '_true_parallel_active'):
        return
    
    print("[ParallelAnything] Cleaning up parallel model...")
    
    # Restore original forward
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        try:
            delattr(model, '_original_forward')
        except:
            pass
    
    # Clear replicas explicitly
    if hasattr(model, '_parallel_replicas'):
        replicas = model._parallel_replicas
        for dev_name, replica in replicas.items():
            try:
                # Move to CPU first to free GPU memory
                replica.cpu()
                # Clear attributes that might hold tensors
                for name, module in replica.named_modules():
                    clear_flux_caches(module)
            except:
                pass
        # Delete the dict
        try:
            delattr(model, '_parallel_replicas')
        except:
            pass
    
    # Remove other attributes
    for attr in ['_true_parallel_active', '_parallel_devices', 'forward_orig_backup']:
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except:
                pass
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        comfy.model_management.soft_empty_cache()

def extract_model_config(model):
    """Extract initialization config from model instance with FLUX-specific handling."""
    config = {}
    possible_attrs = [
        'in_channels', 'out_channels', 'vec_in_dim', 'context_in_dim', 'hidden_size',
        'mlp_ratio', 'num_heads', 'depth', 'depth_single_blocks', 'depth_single',
        'axes_dim', 'theta', 'patch_size', 'qkv_bias', 'guidance_embed',
        'txt_ids_dim', 'img_ids_dim', 'num_res_blocks', 'attention_resolutions',
        'dropout', 'channel_mult', 'num_classes', 'use_checkpoint', 'num_heads_upsample',
        'use_scale_shift_norm', 'resblock_updown', 'use_new_attention_order',
        'adm_in_channels', 'num_noises', 'context_dim', 'n_heads', 'd_head',
        'transformer_depth', 'model_channels', 'max_depth',
    ]
    
    for attr in possible_attrs:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if not callable(val) and not isinstance(val, (nn.Module, torch.nn.Parameter, torch.Tensor)):
                    config[attr] = val
            except Exception:
                pass
    
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
                    except:
                        pass
            else:
                try:
                    params_dict = vars(params)
                    if params_dict:
                        config.update({k: v for k, v in params_dict.items() 
                                     if not isinstance(v, (torch.Tensor, nn.Module))})
                except:
                    pass
        except Exception:
            pass
    
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
        except:
            pass
    
    if hasattr(model, 'unet_config') and isinstance(model.unet_config, dict):
        config.update({k: v for k, v in model.unet_config.items() 
                      if not isinstance(v, (torch.Tensor, nn.Module))})
    
    clean_config = {}
    for k, v in config.items():
        if v is not None and not isinstance(v, (torch.Tensor, nn.Module)):
            clean_config[k] = v
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
                        field_values[field_info.name] = type(val)(
                            clone_dataclass_or_object(v) if is_dataclass(v) 
                            else (v.clone() if isinstance(v, torch.Tensor) else v)
                            for v in val
                        )
                    else:
                        field_values[field_info.name] = copy.deepcopy(val)
                except:
                    field_values[field_info.name] = getattr(obj, field_info.name)
            return obj.__class__(**field_values)
        except:
            pass
    try:
        return copy.deepcopy(obj)
    except:
        return obj

def safe_getattr(obj, attr, default=None):
    """Safely get attribute, returning default if it doesn't exist."""
    return getattr(obj, attr, default)

def clone_module_simple(module, target_device):
    """Simple module cloning that handles Parameters and FLUX-specific attributes."""
    if module is None:
        return None
    module_class = module.__class__
    
    # Try standard reconstruction for basic layers with SAFE attribute access
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        try:
            in_channels = safe_getattr(module, 'in_features') or safe_getattr(module, 'in_channels', 0)
            out_channels = safe_getattr(module, 'out_features') or safe_getattr(module, 'out_channels', 0)
            has_bias = safe_getattr(module, 'bias') is not None
            weight_dtype = safe_getattr(safe_getattr(module, 'weight'), 'dtype', torch.float32)
            
            if isinstance(module, nn.Linear):
                new_mod = nn.Linear(
                    in_channels, out_channels, bias=has_bias,
                    device=target_device, dtype=weight_dtype
                )
            elif isinstance(module, nn.Conv2d):
                kernel_size = safe_getattr(module, 'kernel_size', 1)
                stride = safe_getattr(module, 'stride', 1)
                padding = safe_getattr(module, 'padding', 0)
                dilation = safe_getattr(module, 'dilation', 1)
                groups = safe_getattr(module, 'groups', 1)
                padding_mode = safe_getattr(module, 'padding_mode', 'zeros')
                new_mod = nn.Conv2d(
                    in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                    bias=has_bias, padding_mode=padding_mode,
                    device=target_device, dtype=weight_dtype
                )
            elif isinstance(module, nn.Conv1d):
                kernel_size = safe_getattr(module, 'kernel_size', 1)
                stride = safe_getattr(module, 'stride', 1)
                padding = safe_getattr(module, 'padding', 0)
                dilation = safe_getattr(module, 'dilation', 1)
                groups = safe_getattr(module, 'groups', 1)
                padding_mode = safe_getattr(module, 'padding_mode', 'zeros')
                new_mod = nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                    bias=has_bias, padding_mode=padding_mode,
                    device=target_device, dtype=weight_dtype
                )
            
            with torch.no_grad():
                if hasattr(module, 'weight') and module.weight is not None:
                    new_mod.weight.copy_(module.weight)
                if has_bias and hasattr(module, 'bias') and module.bias is not None:
                    new_mod.bias.copy_(module.bias)
            return new_mod
        except Exception as e:
            print(f"[ParallelAnything] Warning: Failed to reconstruct {module_class}: {e}")
    
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        try:
            normalized_shape = safe_getattr(module, 'normalized_shape')
            num_features = safe_getattr(module, 'num_features', 0)
            num_groups = safe_getattr(module, 'num_groups', 1)
            num_channels = safe_getattr(module, 'num_channels', 0)
            eps = safe_getattr(module, 'eps', 1e-5)
            elementwise_affine = safe_getattr(module, 'elementwise_affine', True)
            momentum = safe_getattr(module, 'momentum', 0.1)
            affine = safe_getattr(module, 'affine', True)
            track_running_stats = safe_getattr(module, 'track_running_stats', True)
            weight_dtype = safe_getattr(safe_getattr(module, 'weight'), 'dtype', torch.float32)
            if weight_dtype is None:
                weight_dtype = torch.float32
            
            if isinstance(module, nn.LayerNorm):
                new_mod = nn.LayerNorm(
                    normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                    device=target_device, dtype=weight_dtype
                )
            elif isinstance(module, nn.BatchNorm2d):
                new_mod = nn.BatchNorm2d(
                    num_features, eps=eps, momentum=momentum, affine=affine,
                    track_running_stats=track_running_stats,
                    device=target_device, dtype=weight_dtype
                )
            elif isinstance(module, nn.GroupNorm):
                new_mod = nn.GroupNorm(
                    num_groups, num_channels, eps=eps, affine=affine,
                    device=target_device, dtype=weight_dtype
                )
            
            with torch.no_grad():
                if hasattr(module, 'weight') and module.weight is not None:
                    new_mod.weight.copy_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    new_mod.bias.copy_(module.bias)
                if hasattr(module, 'running_mean') and safe_getattr(module, 'running_mean') is not None:
                    new_mod.running_mean.copy_(module.running_mean)
                if hasattr(module, 'running_var') and safe_getattr(module, 'running_var') is not None:
                    new_mod.running_var.copy_(module.running_var)
            return new_mod
        except Exception as e:
            print(f"[ParallelAnything] Warning: Failed to reconstruct norm layer {module_class}: {e}")
    
    # Fallback: Create empty instance and manually populate
    try:
        new_mod = module_class.__new__(module_class)
        nn.Module.__init__(new_mod)
        
        # Copy parameters
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                new_param = nn.Parameter(
                    param.clone().detach().to(device=target_device),
                    requires_grad=False
                )
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
        
        # Recursively clone sub-modules
        for name, child in module.named_children():
            if child is not None:
                cloned_child = clone_module_simple(child, target_device)
                new_mod.add_module(name, cloned_child)
        
        # FLUX FIX: Skip batch-dependent cached tensors
        cache_attrs = {
            'img_ids', 'txt_ids', '_img_ids', '_txt_ids', 'cached_img_ids', 'cached_txt_ids',
            'pos_emb', '_pos_emb', 'pos_embed', '_pos_embed', 'freqs_cis', '_freqs_cis',
            'freqs', '_freqs', 'cache', '_cache', 'kv_cache', '_kv_cache', 'attn_bias', '_attn_bias',
        }
        
        for key, value in module.__dict__.items():
            if key not in ['_parameters', '_buffers', '_modules', '_non_persistent_buffers_set',
                          '_backward_pre_hooks', '_backward_hooks', '_forward_pre_hooks', '_forward_hooks',
                          '_state_dict_hooks', '_load_state_dict_pre_hooks', '_extra_state', '_modules_to_load']:
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
                        try:
                            setattr(new_mod, key, copy.copy(value))
                        except:
                            setattr(new_mod, key, copy.deepcopy(value))
                except Exception:
                    pass
        
        new_mod = new_mod.to(target_device)
        return new_mod
    except Exception as e:
        raise RuntimeError(f"Failed to clone module {module_class}: {e}")

def safe_model_clone(source_model, target_device, disable_flash=False):
    """FLUX-safe model cloning with explicit memory cleanup."""
    clear_flux_caches(source_model)
    
    src_device = next(source_model.parameters()).device if next(source_model.parameters(), None) is not None else torch.device('cpu')
    if src_device.type != 'cpu':
        model_cpu = source_model.cpu()
    else:
        model_cpu = source_model
    
    # Force cleanup of any temporary CPU allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        comfy.model_management.soft_empty_cache()
    
    model_class = model_cpu.__class__
    
    try:
        replica = copy.deepcopy(model_cpu)
        replica = replica.to(target_device)
        method_used = "deepcopy"
    except (pickle.PicklingError, AttributeError, TypeError, RuntimeError) as e:
        if any(x in str(e).lower() for x in ["pickle", "can't pickle", "local object", "copy_"]):
            print(f"[ParallelAnything] Deepcopy failed ({str(e)[:80]}), trying config reconstruction...")
        else:
            raise e
        
        try:
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
                    except:
                        config_obj = SimpleNamespace(**config)
                        replica = model_class(config_obj)
                else:
                    raise e
            
            replica.load_state_dict(state_dict, strict=False)
            replica = replica.to(target_device)
            method_used = "config_reconstruction"
        except Exception as recon_error:
            print(f"[ParallelAnything] Config reconstruction failed: {recon_error}")
            print(f"[ParallelAnything] Attempting manual recursive cloning...")
            try:
                replica = clone_module_simple(model_cpu, target_device)
                method_used = "manual_recursive_clone"
            except Exception as manual_error:
                raise RuntimeError(f"All clone methods failed. Last error: {manual_error}")
    
    # Clean up CPU model copy explicitly
    del model_cpu
    gc.collect()
    
    clear_flux_caches(replica)
    replica.eval()
    for param in replica.parameters():
        param.requires_grad = False
    
    for buffer in replica.buffers():
        if buffer.device != target_device:
            buffer.data = buffer.data.to(target_device)
    
    if disable_flash:
        disable_flash_xformers(replica)
    
    print(f"[ParallelAnything] Cloned via {method_used} to {target_device}")
    return replica

class ParallelDevice:
    """Individual device configuration node with auto-detected hardware."""
    
    @classmethod
    def get_available_devices(cls):
        """Dynamically detect all available compute devices."""
        devices = ["cpu"]
        
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                devices.append(f"cuda:{i}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            count = torch.xpu.device_count()
            for i in range(count):
                devices.append(f"xpu:{i}")
        
        try:
            import torch_directml
            dml_count = torch_directml.device_count()
            for i in range(dml_count):
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
                    "default": 50.0, "min": 1.0, "max": 100.0, "step": 1.0,
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
    """Alternative: Parallel layout with dropdowns for 1-4 devices."""
    
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
    """Main execution node with TRUE parallelism via model replication."""
    
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
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "setup_parallel"
    CATEGORY = "utils/hardware"
    
    def setup_parallel(self, model, device_chain, workload_split=True):
        if model is None or not device_chain:
            return (model,)
        
        target_model = model.model.diffusion_model
        
        # Check if already parallelized - cleanup old replicas first
        if hasattr(target_model, "_true_parallel_active"):
            print("[ParallelAnything] Cleaning up previous parallel setup...")
            cleanup_parallel_model(target_model)
            # Also clear any lingering references
            if hasattr(target_model, '_parallel_replicas'):
                delattr(target_model, '_parallel_replicas')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                comfy.model_management.soft_empty_cache()
        
        # Normalize weights
        total_pct = sum(item["percentage"] for item in device_chain)
        if total_pct <= 0:
            return (model,)
        
        device_names = []
        weights = []
        for item in device_chain:
            weights.append(item["percentage"] / total_pct)
            device_names.append(item["device"])
        
        print(f"[ParallelAnything] Setup: {list(zip(device_names, [f'{w*100:.1f}%' for w in weights]))}")
        lead_device = torch.device(device_names[0])
        
        # Check which devices need flash/xformers disabled
        devices_needing_safe_attention = {}
        for dev_name in device_names:
            if not check_sm80_support(dev_name):
                devices_needing_safe_attention[dev_name] = True
                print(f"[ParallelAnything] {dev_name} < SM_80, disabling Flash/xFormers")
        
        # Validate devices
        for dev in device_names:
            try:
                torch.device(dev)
            except:
                print(f"[ParallelAnything] Invalid device: {dev}")
                return (model,)
        
        # Store original device to restore later if needed
        try:
            original_device = next(target_model.parameters()).device
        except StopIteration:
            original_device = torch.device('cpu')
        
        # Create replicas (N× VRAM)
        replicas = {}
        try:
            print(f"[ParallelAnything] Cloning to {len(device_names)} devices... (safe mode)")
            clear_flux_caches(target_model)
            
            # Move original to CPU to free GPU memory during cloning
            if original_device.type == 'cuda':
                print(f"[ParallelAnything] Moving model to CPU for safe cloning...")
                target_model = target_model.cpu()
                torch.cuda.empty_cache()
                comfy.model_management.soft_empty_cache()
            
            for dev_name in device_names:
                dev = torch.device(dev_name)
                need_safe = dev_name in devices_needing_safe_attention
                
                try:
                    replica = safe_model_clone(target_model, dev, disable_flash=need_safe)
                    clear_flux_caches(replica)
                    replicas[dev_name] = replica
                    print(f"[ParallelAnything] ✓ {dev_name}" + (" (Safe mode)" if need_safe else ""))
                    
                    # Aggressive cleanup between clones to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        comfy.model_management.soft_empty_cache()
                        
                except Exception as e:
                    print(f"[ParallelAnything] Error cloning to {dev_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Optionally restore original model to its original device (keep it usable)
            try:
                if original_device.type == 'cuda':
                    print(f"[ParallelAnything] Restoring original model to {original_device}")
                    target_model = target_model.to(original_device)
            except Exception as e:
                print(f"[ParallelAnything] Warning: Could not restore original model device: {e}")
                
        except RuntimeError as e:
            print(f"[ParallelAnything] VRAM Error: {e}")
            # Try to restore original device on error
            try:
                if original_device.type == 'cuda':
                    target_model = target_model.to(original_device)
            except:
                pass
            return (model,)
        
        # Store references for closure
        replicas_ref = replicas
        devices_ref = device_names
        weights_ref = weights
        lead_ref = lead_device
        
        def split_kwargs(kwargs, split_sizes, device):
            """FLUX FIX: Split any tensor kwargs that have batch dimension."""
            split_kwargs_list = [{} for _ in range(len(split_sizes))]
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == sum(split_sizes):
                    # This is a batch tensor, split it
                    chunks = torch.split(value, split_sizes, dim=0)
                    for i, chunk in enumerate(chunks):
                        split_kwargs_list[i][key] = chunk.to(device)
                else:
                    # Not a batch tensor or doesn't match batch size, broadcast to all
                    for i in range(len(split_sizes)):
                        split_kwargs_list[i][key] = value
            return split_kwargs_list
        
        def parallel_forward(self, x, timesteps, context=None, **kwargs):
            batch_size = x.shape[0]
            
            if batch_size < len(devices_ref) or not workload_split:
                with torch.no_grad():
                    return replicas_ref[devices_ref[0]](x, timesteps, context=context, **kwargs)
            
            # Calculate splits
            split_sizes = [max(1, int(batch_size * w)) for w in weights_ref]
            split_sizes[-1] = batch_size - sum(split_sizes[:-1])
            
            active = []
            for idx, (dev_name, size) in enumerate(zip(devices_ref, split_sizes)):
                if size > 0:
                    active.append({
                        'idx': idx,
                        'dev_name': dev_name,
                        'device': torch.device(dev_name),
                        'replica': replicas_ref[dev_name],
                        'size': size,
                    })
            
            if len(active) == 1:
                with torch.no_grad():
                    return active[0]['replica'](x, timesteps, context=context, **kwargs)
            
            # Split tensors
            x_chunks = torch.split(x, [a['size'] for a in active], dim=0)
            t_chunks = torch.split(timesteps, [a['size'] for a in active], dim=0)
            c_chunks = torch.split(context, [a['size'] for a in active], dim=0) if context is not None else [None] * len(active)
            
            # FLUX FIX: Split kwargs tensors (guidance, y, etc.)
            kwargs_chunks = split_kwargs(kwargs, [a['size'] for a in active], lead_ref)
            
            results = [None] * len(active)
            
            def worker(task_idx):
                task = active[task_idx]
                dev = task['device']
                replica = task['replica']
                
                try:
                    if dev.type == 'cuda':
                        with torch.cuda.device(dev):
                            torch.cuda.synchronize(dev)
                            x_in = x_chunks[task_idx].to(dev, non_blocking=True)
                            t_in = t_chunks[task_idx].to(dev, non_blocking=True)
                            c_in = c_chunks[task_idx].to(dev, non_blocking=True) if c_chunks[task_idx] is not None else None
                            
                            # Get the split kwargs for this chunk
                            k_in = kwargs_chunks[task_idx]
                            # Move any remaining tensors to device
                            for k, v in k_in.items():
                                if isinstance(v, torch.Tensor) and v.device != dev:
                                    k_in[k] = v.to(dev)
                            
                            with torch.no_grad():
                                out = replica(x_in, t_in, context=c_in, **k_in)
                            torch.cuda.synchronize(dev)
                            return task_idx, out.to(lead_ref, non_blocking=False)
                            
                    elif dev.type == 'xpu':
                        with torch.xpu.device(dev):
                            torch.xpu.synchronize(dev)
                            x_in = x_chunks[task_idx].to(dev, non_blocking=True)
                            t_in = t_chunks[task_idx].to(dev, non_blocking=True)
                            c_in = c_chunks[task_idx].to(dev, non_blocking=True) if c_chunks[task_idx] is not None else None
                            
                            k_in = kwargs_chunks[task_idx]
                            for k, v in k_in.items():
                                if isinstance(v, torch.Tensor) and v.device != dev:
                                    k_in[k] = v.to(dev)
                            
                            with torch.no_grad():
                                out = replica(x_in, t_in, context=c_in, **k_in)
                            torch.xpu.synchronize(dev)
                            return task_idx, out.to(lead_ref, non_blocking=False)
                            
                    else:
                        # CPU/MPS
                        x_in = x_chunks[task_idx].to(dev)
                        t_in = t_chunks[task_idx].to(dev)
                        c_in = c_chunks[task_idx].to(dev) if c_chunks[task_idx] is not None else None
                        
                        k_in = kwargs_chunks[task_idx]
                        for k, v in k_in.items():
                            if isinstance(v, torch.Tensor) and v.device != dev:
                                k_in[k] = v.to(dev)
                        
                        with torch.no_grad():
                            out = replica(x_in, t_in, context=c_in, **k_in)
                        return task_idx, out.to(lead_ref)
                        
                except Exception as e:
                    return task_idx, e
            
            # TRUE PARALLEL EXECUTION
            with ThreadPoolExecutor(max_workers=len(active)) as executor:
                futures = [executor.submit(worker, i) for i in range(len(active))]
                for future in as_completed(futures):
                    idx, result = future.result()
                    if isinstance(result, Exception):
                        raise result
                    results[idx] = result
            
            return torch.cat(results, dim=0)
        
        # Replace forward method
        target_model._original_forward = target_model.forward
        target_model.forward = types.MethodType(parallel_forward, target_model)
        target_model._true_parallel_active = True
        target_model._parallel_replicas = replicas
        target_model._parallel_devices = device_names
        
        # Register cleanup on model deletion
        weakref.finalize(model, cleanup_parallel_model, target_model)
        
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
