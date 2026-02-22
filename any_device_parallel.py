import torch
import torch.nn as nn
import types
import copy
import gc
import weakref
import threading
from types import SimpleNamespace
from dataclasses import dataclass, fields, is_dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import comfy.model_management

# ----------------------------------------------------------------------------
# Thread-local state for pipeline mode
# ----------------------------------------------------------------------------
_pipeline_state = threading.local()

def get_pipeline_mode():
    return getattr(_pipeline_state, 'active', False)

def set_pipeline_mode(active):
    _pipeline_state.active = active

class ParallelBlock(nn.Module):
    """
    A wrapper that handles dual-mode execution:
    1. Data Parallel (Batch > 1): Runs the local block immediately.
    2. Pipeline Parallel (Batch = 1): Moves data to assigned device/replica.
    """
    def __init__(self, local_block, block_idx, owner_device, peers, is_last_block, lead_device):
        super().__init__()
        self.local_block = local_block
        self.block_idx = block_idx
        self.owner_device = owner_device
        self.peers = peers
        self.is_last_block = is_last_block
        self.lead_device = lead_device
        self.owner_block = self.peers.get(str(owner_device), local_block)

    def _move_tensor(self, x, device):
        """Recursively move tensors, lists, tuples, dicts, and dataclasses to target device."""
        if isinstance(x, torch.Tensor):
            if x.device != device:
                return x.to(device, non_blocking=True)
            return x
        elif isinstance(x, (list, tuple)):
            return type(x)(self._move_tensor(item, device) for item in x)
        elif isinstance(x, dict):
            return {k: self._move_tensor(v, device) for k, v in x.items()}
        elif is_dataclass(x):
            # Shallow copy dataclass and move its tensor fields
            new_obj = copy.copy(x)
            for field in fields(x):
                val = getattr(x, field.name)
                setattr(new_obj, field.name, self._move_tensor(val, device))
            return new_obj
        return x

    def _move_args(self, args, device):
        return tuple(self._move_tensor(a, device) for a in args)

    def _move_kwargs(self, kwargs, device):
        return {k: self._move_tensor(v, device) for k, v in kwargs.items()}

    def forward(self, *args, **kwargs):
        if not get_pipeline_mode():
            # Standard execution for Data Parallel (Batch > 1)
            return self.local_block(*args, **kwargs)

        target_device = self.owner_device

        # --- CRITICAL FIX ---
        # Do NOT check args[0].device to skip moving.
        # Main input might be on target_device (from prev block),
        # but auxiliary inputs (timesteps, context) might still be on Lead Device.
        # We must ensure ALL inputs are on target_device.
        args_moved = self._move_args(args, target_device)
        kwargs_moved = self._move_kwargs(kwargs, target_device)

        # Execute on target replica
        out = self.owner_block(*args_moved, **kwargs_moved)

        if self.is_last_block:
            # Return data to Lead Device for final layers/output processing
            out = self._move_tensor(out, self.lead_device)

        return out

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def is_float8_dtype(dtype):
    if dtype is None:
        return False
    dtype_name = str(dtype)
    fp8_types = ['float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz', 'float8']
    return any(t in dtype_name for t in fp8_types)

def check_sm80_support(device_name):
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

def device_supports_float8(device):
    if isinstance(device, str):
        if not device.startswith("cuda"):
            return False
        device = torch.device(device)
    if device.type != 'cuda':
        return False
    try:
        cap = torch.cuda.get_device_capability(device)
        major, minor = cap
        return (major, minor) >= (9, 0)
    except:
        return False

def disable_flash_xformers(model):
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
    
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['attn', 'attention', 'transformer']):
            for attr in ['use_xformers', 'use_flash_attention', 'use_flash_attention_2', '_use_memory_efficient_attention', 'enable_flash', 'enable_xformers']:
                if hasattr(module, attr):
                    try:
                        setattr(module, attr, False)
                    except AttributeError:
                        pass
            if hasattr(module, 'set_processor'):
                try:
                    from diffusers.models.attention_processor import Attention
                    module.set_processor(Attention())
                except Exception:
                    pass

def clear_flux_caches(model):
    cache_attrs = [
        'img_ids', 'txt_ids', '_img_ids', '_txt_ids', 'cached_img_ids', 'cached_txt_ids',
        'pos_emb', '_pos_emb', 'pos_embed', '_pos_embed', 'cached_pos_emb',
        'rope', '_rope', 'freqs_cis', '_freqs_cis', 'freqs', '_freqs',
        'cache', '_cache', 'kv_cache', '_kv_cache', 'attn_bias', '_attn_bias',
        'rope_cache', '_rope_cache', 'freqs_cis_cache', '_freqs_cis_cache',
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
    
    clear_attrs(model)
    for name, module in model.named_modules():
        clear_attrs(module, name)
    
    if cleared_count > 0:
        print(f"[ParallelAnything] Cleared {cleared_count} cached tensors")

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(f'cuda:{i}'):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except:
                pass
    comfy.model_management.soft_empty_cache()

def cleanup_parallel_model(model_ref):
    model = model_ref() if isinstance(model_ref, weakref.ref) else model_ref
    if model is None:
        return
    
    if not getattr(model, '_true_parallel_active', False):
        return
    
    print("[ParallelAnything] Cleaning up parallel model...")
    should_purge_cache = getattr(model, '_parallel_purge_cache', True)
    should_purge_models = getattr(model, '_parallel_purge_models', False)
    
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
                if hasattr(replica, 'cpu'):
                    replica.cpu()
                clear_flux_caches(replica)
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
    
    # Clear other attributes
    for attr in ['_true_parallel_active', '_parallel_devices', '_parallel_streams', 
                 '_parallel_weights', '_auto_vram_balance', '_parallel_purge_cache', 
                 '_parallel_purge_models']:
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass
    
    if should_purge_models:
        try:
            print("[ParallelAnything] Purging models from VRAM...")
            comfy.model_management.unload_all_models()
        except Exception as e:
            print(f"[ParallelAnything] Warning: Could not unload models: {e}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if should_purge_cache:
        try:
            comfy.model_management.soft_empty_cache()
        except:
            pass
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(f'cuda:{i}'):
                        torch.cuda.empty_cache()
                except:
                    pass

def extract_model_config(model):
    config = {}
    possible_attrs = [
        'in_channels', 'out_channels', 'vec_in_dim', 'context_in_dim', 'hidden_size',
        'mlp_ratio', 'num_heads', 'depth', 'depth_single_blocks', 'depth_single',
        'axes_dim', 'theta', 'patch_size', 'qkv_bias', 'guidance_embed',
        'txt_ids_dim', 'img_ids_dim', 'num_res_blocks', 'attention_resolutions',
        'dropout', 'channel_mult', 'num_classes', 'use_checkpoint', 'num_heads_upsample',
        'use_scale_shift_norm', 'resblock_updown', 'use_new_attention_order',
        'adm_in_channels', 'num_noises', 'context_dim', 'n_heads', 'd_head',
        'transformer_depth', 'model_channels', 'max_depth', 'num_frames',
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
                    config.update({k: v for k, v in params_dict.items() if not isinstance(v, (torch.Tensor, nn.Module))})
        except Exception:
            pass
    
    if hasattr(model, 'config'):
        try:
            cfg = model.config
            if isinstance(cfg, dict):
                config.update({k: v for k, v in cfg.items() if not isinstance(v, (torch.Tensor, nn.Module))})
            else:
                cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_') and not callable(v) and not isinstance(v, (torch.Tensor, nn.Module))}
                config.update(cfg_dict)
        except Exception:
            pass
    
    if hasattr(model, 'unet_config') and isinstance(model.unet_config, dict):
        config.update({k: v for k, v in model.unet_config.items() if not isinstance(v, (torch.Tensor, nn.Module))})
    
    clean_config = {}
    for k, v in config.items():
        if v is not None and not isinstance(v, (torch.Tensor, nn.Module)):
            try:
                copy.deepcopy(v)
                clean_config[k] = v
            except Exception:
                pass
    
    return clean_config

def clone_dataclass_or_object(obj, target_device=None):
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
                    # Fallback to direct reference if deep copy fails
                    field_values[field_info.name] = getattr(obj, field_info.name)
            return obj.__class__(**field_values)
        except Exception:
            pass
    
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj

def safe_getattr(obj, attr, default=None):
    return getattr(obj, attr, default)

def clone_module_simple(module, target_device):
    if module is None:
        return None
    
    module_class = module.__class__
    
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        try:
            weight = safe_getattr(module, 'weight', None)
            weight_dtype = weight.dtype if weight is not None else torch.float32
            has_bias = safe_getattr(module, 'bias', None) is not None
            
            supports_fp8 = device_supports_float8(target_device)
            if not supports_fp8 and is_float8_dtype(weight_dtype):
                weight_dtype = torch.float16
            
            if isinstance(module, nn.Linear):
                new_mod = nn.Linear(
                    safe_getattr(module, 'in_features'),
                    safe_getattr(module, 'out_features'),
                    bias=has_bias,
                    device=target_device,
                    dtype=weight_dtype
                )
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
                    if not supports_fp8 and is_float8_dtype(weight.dtype):
                        weight = weight.half()
                    new_mod.weight.copy_(weight)
                if has_bias and safe_getattr(module, 'bias') is not None:
                    new_mod.bias.copy_(module.bias)
            return new_mod
        except Exception as e:
            print(f"[ParallelAnything] Warning: Failed to reconstruct {module_class}: {e}")
    
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        try:
            weight = safe_getattr(module, 'weight', None)
            weight_dtype = weight.dtype if weight is not None else torch.float32
            supports_fp8 = device_supports_float8(target_device)
            if not supports_fp8 and is_float8_dtype(weight_dtype):
                weight_dtype = torch.float16
            
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
                    if not supports_fp8 and is_float8_dtype(weight.dtype):
                        weight = weight.half()
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
    
    try:
        new_mod = module_class.__new__(module_class)
        nn.Module.__init__(new_mod)
        supports_fp8 = device_supports_float8(target_device)
        
        # Copy parameters
        for name, param in module.named_parameters(recurse=False):
            if param is not None:
                new_param_data = param.clone().detach()
                if not supports_fp8 and is_float8_dtype(new_param_data.dtype):
                    new_param_data = new_param_data.half()
                new_param = nn.Parameter(new_param_data.to(device=target_device), requires_grad=False)
                new_mod.register_parameter(name, new_param)
            else:
                new_mod.register_parameter(name, None)
        
        # Copy buffers
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                new_buffer_data = buffer.clone().detach()
                if not supports_fp8 and is_float8_dtype(new_buffer_data.dtype):
                    new_buffer_data = new_buffer_data.half()
                new_buffer = new_buffer_data.to(device=target_device)
                new_mod.register_buffer(name, new_buffer)
            else:
                new_mod.register_buffer(name, None)
        
        # Recursively clone children
        for name, child in module.named_children():
            if child is not None:
                cloned_child = clone_module_simple(child, target_device)
                new_mod.add_module(name, cloned_child)
        
        # Copy other attributes
        excluded_attrs = {
            '_parameters', '_buffers', '_modules', '_non_persistent_buffers_set',
            '_backward_pre_hooks', '_backward_hooks', '_forward_pre_hooks',
            '_forward_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks',
            '_extra_state', '_modules_to_load'
        }
        cache_attrs = {
            'img_ids', 'txt_ids', '_img_ids', '_txt_ids', 'cached_img_ids', 'cached_txt_ids',
            'pos_emb', '_pos_emb', 'pos_embed', '_pos_embed', 'freqs_cis', '_freqs_cis',
            'freqs', '_freqs', 'cache', '_cache', 'kv_cache', '_kv_cache', 'attn_bias', '_attn_bias'
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
                        tensor_val = value.clone().detach()
                        if not supports_fp8 and is_float8_dtype(tensor_val.dtype):
                            tensor_val = tensor_val.half()
                        setattr(new_mod, key, tensor_val.to(target_device))
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    if key in cache_attrs:
                        setattr(new_mod, key, None)
                    else:
                        converted = []
                        for t in value:
                            t_conv = t.clone().detach()
                            if not supports_fp8 and is_float8_dtype(t_conv.dtype):
                                t_conv = t_conv.half()
                            converted.append(t_conv.to(target_device))
                        setattr(new_mod, key, type(value)(converted))
                else:
                    setattr(new_mod, key, copy.deepcopy(value))
            except Exception:
                pass
        
        return new_mod
    except Exception as e:
        raise RuntimeError(f"Failed to clone module {module_class}: {e}")

def safe_model_clone(source_model, target_device, disable_flash=False):
    clear_flux_caches(source_model)
    
    try:
        src_device = next(source_model.parameters()).device
    except StopIteration:
        src_device = torch.device('cpu')
    
    if str(src_device) == str(target_device):
        print(f"[ParallelAnything] Model already on {target_device}, using reference...")
        clear_flux_caches(source_model)
        return source_model
    
    # Move to CPU first if on GPU
    if src_device.type == 'cuda':
        model_cpu = source_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            comfy.model_management.soft_empty_cache()
    else:
        model_cpu = source_model
    
    model_class = model_cpu.__class__
    replica = None
    method_used = "unknown"
    
    try:
        print(f"[ParallelAnything] Attempting efficient clone to {target_device}...")
        config = extract_model_config(model_cpu)
        state_dict = model_cpu.state_dict()
        
        if not config:
            raise RuntimeError("Could not extract model config")
        
        try:
            replica = model_class(**config)
            replica = replica.to(target_device, non_blocking=False)
        except TypeError as e:
            if "missing" in str(e) and "required" in str(e):
                try:
                    replica = model_class(config)
                    replica = replica.to(target_device, non_blocking=False)
                except Exception:
                    config_obj = SimpleNamespace(**config)
                    replica = model_class(config_obj)
                    replica = replica.to(target_device, non_blocking=False)
            else:
                raise
        
        print(f"[ParallelAnything] Loading parameters incrementally...")
        supports_fp8 = device_supports_float8(target_device)
        
        with torch.no_grad():
            for key in list(state_dict.keys()):
                try:
                    param = state_dict[key]
                    if '.' in key:
                        parts = key.split('.')
                        module = replica
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        target_param = getattr(module, parts[-1])
                    else:
                        target_param = getattr(replica, key)
                    
                    if isinstance(target_param, (torch.nn.Parameter, torch.Tensor)):
                        param_data = param.to(target_device, non_blocking=False)
                        if is_float8_dtype(param_data.dtype) and not supports_fp8:
                            param_data = param_data.half()
                        target_param.data.copy_(param_data)
                    
                    del state_dict[key]
                    del param
                    if len(state_dict) % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ParallelAnything] Warning: Could not load {key}: {e}")
        
        del state_dict
        if src_device.type != 'cuda':
            del model_cpu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        method_used = "incremental_gpu"
        
    except Exception as recon_error:
        print(f"[ParallelAnything] Config reconstruction failed: {recon_error}")
        print(f"[ParallelAnything] Attempting manual recursive cloning...")
        replica = clone_module_simple(model_cpu, target_device)
        method_used = "manual_recursive_clone"
        if src_device.type != 'cuda':
            del model_cpu
        gc.collect()
    
    if replica is None:
        raise RuntimeError("All clone methods failed")
    
    clear_flux_caches(replica)
    
    if not device_supports_float8(target_device):
        print(f"[ParallelAnything] Converting FP8 weights to FP16 for {target_device}...")
        def convert_fp8_recursive(m):
            for name, param in m.named_parameters():
                if is_float8_dtype(param.dtype):
                    param.data = param.data.half()
            for name, buffer in m.named_buffers():
                if is_float8_dtype(buffer.dtype):
                    buffer.data = buffer.data.half()
            for child in m.children():
                convert_fp8_recursive(child)
        convert_fp8_recursive(replica)
    
    if hasattr(replica, 'gradient_checkpointing'):
        replica.gradient_checkpointing = False
    if hasattr(replica, '_gradient_checkpointing_func'):
        replica._gradient_checkpointing_func = None
    if hasattr(replica, 'hooks'):
        replica.hooks = []
    if hasattr(replica, '_hf_hook'):
        replica._hf_hook = None
    
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

def get_free_vram(device_name):
    try:
        if device_name.startswith("cuda"):
            idx = int(device_name.split(":")[-1])
            torch.cuda.set_device(idx)
            free_memory = (
                torch.cuda.get_device_properties(idx).total_memory - torch.cuda.memory_allocated(idx)
            )
            return free_memory / (1024 ** 2)
    except Exception:
        pass
    return 0

def auto_split_batch(batch_size, devices, weights):
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
    
    adjusted_weights = []
    for i, (w, vram) in enumerate(zip(weights, vram_avail)):
        if vram > 0:
            vram_weight = vram / total_vram
            adjusted = 0.7 * w + 0.3 * vram_weight
        else:
            adjusted = w
        adjusted_weights.append(adjusted)
    
    total = sum(adjusted_weights)
    adjusted_weights = [w/total for w in adjusted_weights]
    
    split_sizes = [max(1, int(batch_size * w)) for w in adjusted_weights]
    split_sizes[-1] = batch_size - sum(split_sizes[:-1])
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
                    "tooltip": "Percentage of batch (or layers for batch=1) to process on this device"
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
                "purge_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Purge CUDA cache when cleaning up parallel resources"
                }),
                "purge_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload all models from VRAM when cleaning up (aggressive memory clearing)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "setup_parallel"
    CATEGORY = "utils/hardware"

    def setup_parallel(self, model, device_chain, workload_split=True, auto_vram_balance=False, purge_cache=True, purge_models=False):
        if model is None or not device_chain:
            return (model,)
        
        # Extract the diffusion model from ComfyUI model wrapper
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            target_model = model.model.diffusion_model
            model_wrapper = model
        elif hasattr(model, 'diffusion_model'):
            target_model = model.diffusion_model
            model_wrapper = None
        else:
            target_model = model
            model_wrapper = None
        
        # Check if ComfyUI model has patches (LoRAs) that need to be applied
        # ComfyUI stores patches in model.patches (list of tuples) or model.model_patcher.patches
        has_lora_patches = False
        if model_wrapper is not None:
            # Check various locations where ComfyUI might store patches
            if hasattr(model_wrapper, 'patches') and len(model_wrapper.patches) > 0:
                has_lora_patches = True
                print(f"[ParallelAnything] Found {len(model_wrapper.patches)} patches in model.patches")
            elif hasattr(model_wrapper, 'model_patcher') and hasattr(model_wrapper.model_patcher, 'patches'):
                if len(model_wrapper.model_patcher.patches) > 0:
                    has_lora_patches = True
                    print(f"[ParallelAnything] Found {len(model_wrapper.model_patcher.patches)} patches in model_patcher.patches")
        
        # Apply patches if found - this modifies the diffusion_model weights in-place
        if has_lora_patches:
            try:
                print("[ParallelAnything] Applying LoRA/Custom patches before cloning...")
                # Apply patches to the model weights
                if hasattr(model_wrapper, 'patch_model'):
                    model_wrapper.patch_model(device_to=comfy.model_management.get_torch_device())
                    print("[ParallelAnything] Patches applied successfully")
                elif hasattr(model_wrapper, 'model_patcher') and hasattr(model_wrapper.model_patcher, 'patch_model'):
                    model_wrapper.model_patcher.patch_model(device_to=comfy.model_management.get_torch_device())
                    print("[ParallelAnything] Patches applied via model_patcher")
            except Exception as e:
                print(f"[ParallelAnything] Warning: Could not apply patches: {e}")
                has_lora_patches = False
        
        # Cleanup previous parallel setup if exists
        if hasattr(target_model, "_true_parallel_active"):
            print("[ParallelAnything] Cleaning up previous parallel setup...")
            cleanup_parallel_model(target_model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                comfy.model_management.soft_empty_cache()
        
        print("[ParallelAnything] Freeing ComfyUI model cache for cloning...")
        comfy.model_management.unload_all_models()
        aggressive_cleanup()
        
        total_pct = sum(item["percentage"] for item in device_chain)
        if total_pct <= 0:
            return (model,)
        
        device_names = []
        weights = []
        for item in device_chain:
            weights.append(item["percentage"] / total_pct)
            device_names.append(item["device"])
        
        print(f"[ParallelAnything] Setup: {list(zip(device_names, [f'{w*100:.1f}%' for w in weights]))}")
        
        devices_needing_safe_attention = {}
        for dev_name in device_names:
            if not check_sm80_support(dev_name):
                devices_needing_safe_attention[dev_name] = True
                print(f"[ParallelAnything] {dev_name} < SM_80, disabling Flash/xFormers")
        
        for dev in device_names:
            try:
                torch.device(dev)
            except Exception:
                print(f"[ParallelAnything] Invalid device: {dev}")
                return (model,)
        
        try:
            original_device = next(target_model.parameters()).device
            original_device_str = str(original_device)
        except StopIteration:
            original_device = torch.device('cpu')
            original_device_str = "cpu"
        
        replicas = {}
        streams = {}
        successful_devices = []
        successful_weights = []
        
        try:
            print(f"[ParallelAnything] Cloning to {len(device_names)} devices...")
            clear_flux_caches(target_model)
            
            needs_cpu_transition = original_device.type == 'cuda' and original_device_str not in device_names
            if needs_cpu_transition:
                print(f"[ParallelAnything] Moving model to CPU for safe cloning...")
                target_model = target_model.cpu()
                aggressive_cleanup()
            
            for i, dev_name in enumerate(device_names):
                dev = torch.device(dev_name)
                need_safe = dev_name in devices_needing_safe_attention
                
                # Determine if this is the original device
                is_original_device = (dev_name == original_device_str and not needs_cpu_transition)
                
                # CRITICAL FIX: If we have LoRA patches, we MUST clone even the original device
                # because ComfyUI will manage the original model's weights separately
                if is_original_device and not has_lora_patches:
                    print(f"[ParallelAnything] Using original model for {dev_name} (skipping clone)")
                    replicas[dev_name] = target_model
                    if dev.type == 'cuda':
                        streams[dev_name] = torch.cuda.Stream(dev)
                    successful_devices.append(dev_name)
                    successful_weights.append(weights[i])
                    continue
                
                try:
                    if is_original_device and has_lora_patches:
                        print(f"[ParallelAnything] Cloning to {dev_name} (preserving LoRA weights)...")
                    else:
                        print(f"[ParallelAnything] Cloning to {dev_name} ({i+1}/{len(device_names)})...")
                    
                    if dev.type == 'cuda':
                        idx = dev.index if dev.index is not None else int(dev_name.split(":")[-1])
                        torch.cuda.synchronize(idx)
                        free_mem = torch.cuda.mem_get_info(idx)[0] / 1024**3
                        print(f"[ParallelAnything] VRAM available on {dev_name}: {free_mem:.2f}GB")
                    
                    replica = safe_model_clone(target_model, dev, disable_flash=need_safe)
                    clear_flux_caches(replica)
                    replicas[dev_name] = replica
                    
                    if dev.type == 'cuda':
                        streams[dev_name] = torch.cuda.Stream(dev)
                    
                    status_msg = f"[ParallelAnything] ✓ {dev_name}"
                    if need_safe:
                        status_msg += " (Safe mode)"
                    if has_lora_patches:
                        status_msg += " (LoRA active)"
                    print(status_msg)
                    
                    successful_devices.append(dev_name)
                    successful_weights.append(weights[i])
                    aggressive_cleanup()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[ParallelAnything] OOM cloning to {dev_name}, skipping...")
                        aggressive_cleanup()
                        continue
                    else:
                        raise
            
            if len(successful_devices) < len(device_names):
                if len(successful_devices) == 0:
                    raise RuntimeError("No devices available for parallel processing - all OOM")
                print(f"[ParallelAnything] Reducing to {len(successful_devices)} devices due to OOM")
                device_names = successful_devices
                total_weight = sum(successful_weights)
                weights = [w/total_weight for w in successful_weights]
            
            # Restore original to its device if we moved it to CPU and it's not in our replicas
            if needs_cpu_transition and original_device_str not in replicas:
                try:
                    print(f"[ParallelAnything] Restoring original model to {original_device}")
                    target_model = target_model.to(original_device)
                except Exception as e:
                    print(f"[ParallelAnything] Warning: Could not restore original model: {e}")
                    
        except Exception as e:
            print(f"[ParallelAnything] Error during cloning: {e}")
            import traceback
            traceback.print_exc()
            for dev_name, r in replicas.items():
                try:
                    if r is not target_model:
                        r.cpu()
                        del r
                except Exception:
                    pass
            aggressive_cleanup()
            return (model,)
        
        print("[ParallelAnything] Configuring Model/Pipeline Parallelism for Batch=1...")
        lead_device_name = device_names[0]
        lead_replica = replicas[lead_device_name]
        
        block_lists = ['double_blocks', 'single_blocks', 'transformer_blocks', 'layers']
        for list_name in block_lists:
            if hasattr(lead_replica, list_name):
                local_blocks = getattr(lead_replica, list_name)
                if not isinstance(local_blocks, nn.ModuleList):
                    continue
                num_blocks = len(local_blocks)
                if num_blocks == 0:
                    continue
                
                print(f"[ParallelAnything] Configuring {list_name} ({num_blocks} blocks) for pipeline execution...")
                
                device_assignments = []
                current_block = 0
                for i, weight in enumerate(weights):
                    count = int(round(weight * num_blocks))
                    if i == len(weights) - 1:
                        count = num_blocks - current_block
                    target_dev = torch.device(device_names[i])
                    for _ in range(count):
                        if current_block < num_blocks:
                            device_assignments.append(target_dev)
                            current_block += 1
                
                for idx in range(num_blocks):
                    assigned_dev = device_assignments[idx]
                    peers = {}
                    for d_name, r in replicas.items():
                        if hasattr(r, list_name):
                            peers[d_name] = getattr(r, list_name)[idx]
                    
                    original_block = local_blocks[idx]
                    is_last = (idx == num_blocks - 1)
                    
                    wrapper = ParallelBlock(
                        local_block=original_block,
                        block_idx=idx,
                        owner_device=assigned_dev,
                        peers=peers,
                        is_last_block=is_last,
                        lead_device=torch.device(lead_device_name)
                    )
                    local_blocks[idx] = wrapper
        
        target_model._parallel_purge_cache = purge_cache
        target_model._parallel_purge_models = purge_models
        
        replicas_ref = replicas
        devices_ref = tuple(device_names)
        weights_ref = tuple(weights)
        lead_device = torch.device(device_names[0])
        streams_ref = streams
        auto_balance_ref = auto_vram_balance
        
        def get_batch_size(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0]
            elif isinstance(x, (list, tuple)) and len(x) > 0:
                if isinstance(x[0], torch.Tensor):
                    batch_sizes = [t.shape[0] for t in x if isinstance(t, torch.Tensor)]
                    if batch_sizes:
                        return batch_sizes[0]
                return len(x)
            else:
                return 1
        
        def split_batch(x, split_sizes):
            if isinstance(x, torch.Tensor):
                return torch.split(x, split_sizes, dim=0)
            elif isinstance(x, (list, tuple)):
                split_lists = []
                for t in x:
                    if isinstance(t, torch.Tensor):
                        split_lists.append(torch.split(t, split_sizes, dim=0))
                    else:
                        split_lists.append([t] * len(split_sizes))
                result = []
                for i in range(len(split_sizes)):
                    result.append(type(x)(sl[i] for sl in split_lists))
                return result
            else:
                return [x] * len(split_sizes)
        
        def move_to_device(x, device, non_blocking=False):
            if isinstance(x, torch.Tensor):
                if x.device != torch.device(device):
                    x = x.to(device, non_blocking=non_blocking)
                if is_float8_dtype(x.dtype) and not device_supports_float8(device):
                    x = x.half()
                return x
            elif isinstance(x, (list, tuple)):
                moved = [move_to_device(t, device, non_blocking) for t in x]
                return type(x)(moved)
            else:
                return x
        
        def split_kwargs(kwargs, split_sizes, total_batch):
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
                    for i in range(len(split_sizes)):
                        split_kwargs_list[i][key] = value
            return split_kwargs_list
        
        def concatenate_results(results, dim=0):
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
            try:
                try:
                    batch_size = get_batch_size(x)
                except ValueError as e:
                    print(f"[ParallelAnything] Error: {e}")
                    raise
                
                # Pipeline mode for batch_size == 1
                if batch_size == 1 and workload_split:
                    set_pipeline_mode(True)
                    try:
                        lead_replica = replicas_ref[devices_ref[0]]
                        if hasattr(lead_replica, '_original_forward'):
                            return lead_replica._original_forward(x, timesteps, context=context, **kwargs)
                        else:
                            return lead_replica(x, timesteps, context=context, **kwargs)
                    finally:
                        set_pipeline_mode(False)
                
                # Single device fallback if batch too small or workload_split disabled
                if batch_size < len(devices_ref) or not workload_split:
                    set_pipeline_mode(False)
                    with torch.no_grad():
                        first_replica = replicas_ref[devices_ref[0]]
                        if hasattr(first_replica, '_original_forward'):
                            return first_replica._original_forward(x, timesteps, context=context, **kwargs)
                        else:
                            return first_replica(x, timesteps, context=context, **kwargs)
                
                # Data parallelism for batch_size > 1
                if auto_balance_ref:
                    split_sizes = auto_split_batch(batch_size, list(devices_ref), list(weights_ref))
                else:
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
                            'stream': streams_ref.get(dev_name),
                            'size': size,
                        })
                
                if len(active) == 0:
                    raise RuntimeError("No active devices available")
                
                if len(active) == 1:
                    set_pipeline_mode(False)
                    with torch.no_grad():
                        replica = active[0]['replica']
                        if hasattr(replica, '_original_forward'):
                            return replica._original_forward(x, timesteps, context=context, **kwargs)
                        else:
                            return replica(x, timesteps, context=context, **kwargs)
                
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
                    set_pipeline_mode(False)
                    task = active[task_idx]
                    dev = task['device']
                    replica = task['replica']
                    stream = task['stream']
                    
                    try:
                        if dev.type == 'cuda':
                            torch.cuda.set_device(dev)
                        
                        x_in = move_to_device(x_chunks[task_idx], dev)
                        t_in = move_to_device(t_chunks[task_idx], dev)
                        c_in = move_to_device(c_chunks[task_idx], dev) if c_chunks[task_idx] is not None else None
                        
                        k_in = {}
                        for k, v in kwargs_chunks[task_idx].items():
                            k_in[k] = move_to_device(v, dev, non_blocking=False)
                        
                        if hasattr(replica, '_original_forward'):
                            forward_fn = replica._original_forward
                        else:
                            forward_fn = replica.forward
                        
                        if dev.type == 'cuda' and stream is not None:
                            with torch.cuda.device(dev):
                                with torch.cuda.stream(stream):
                                    torch.cuda.synchronize(dev)
                                    with torch.no_grad():
                                        out = forward_fn(x_in, t_in, context=c_in, **k_in)
                                    torch.cuda.synchronize(dev)
                        elif dev.type == 'cuda':
                            with torch.cuda.device(dev):
                                torch.cuda.synchronize(dev)
                                with torch.no_grad():
                                    out = forward_fn(x_in, t_in, context=c_in, **k_in)
                                torch.cuda.synchronize(dev)
                        elif dev.type == 'xpu':
                            with torch.xpu.device(dev):
                                torch.xpu.synchronize(dev)
                                with torch.no_grad():
                                    out = forward_fn(x_in, t_in, context=c_in, **k_in)
                                torch.xpu.synchronize(dev)
                        else:
                            with torch.no_grad():
                                out = forward_fn(x_in, t_in, context=c_in, **k_in)
                        
                        out = move_to_device(out, lead_device, non_blocking=False)
                        return task_idx, out
                    
                    except Exception as e:
                        return task_idx, e
                
                with ThreadPoolExecutor(max_workers=len(active)) as executor:
                    futures = [executor.submit(worker, i) for i in range(len(active))]
                    for future in as_completed(futures):
                        idx, result = future.result()
                        if isinstance(result, Exception):
                            exceptions.append((active[idx]['dev_name'], result))
                            results[idx] = None
                        else:
                            results[idx] = result
                
                if exceptions:
                    for dev_name, exc in exceptions:
                        print(f"[ParallelAnything] Error on {dev_name}: {exc}")
                    raise exceptions[0][1]
                
                if any(r is None for r in results):
                    missing = [active[i]['dev_name'] for i, r in enumerate(results) if r is None]
                    raise RuntimeError(f"Missing results from devices: {missing}")
                
                return concatenate_results(results, dim=0)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[ParallelAnything] OOM in parallel forward, attempting emergency cleanup...")
                    aggressive_cleanup()
                    print("[ParallelAnything] Falling back to single device processing...")
                    set_pipeline_mode(False)
                    with torch.no_grad():
                        first_replica = replicas_ref[devices_ref[0]]
                        if hasattr(first_replica, '_original_forward'):
                            return first_replica._original_forward(x, timesteps, context=context, **kwargs)
                        else:
                            return first_replica(x, timesteps, context=context, **kwargs)
                else:
                    raise
        
        target_model._original_forward = target_model.forward
        target_model.forward = types.MethodType(parallel_forward, target_model)
        target_model._true_parallel_active = True
        target_model._parallel_replicas = replicas
        target_model._parallel_devices = device_names
        target_model._parallel_streams = streams
        target_model._parallel_weights = weights
        target_model._auto_vram_balance = auto_vram_balance
        
        weakref.finalize(model, cleanup_parallel_model, weakref.ref(target_model))
        
        # Update ComfyUI's model device tracking to point to lead device
        if hasattr(model, 'load_device'):
            model.load_device = lead_device
        elif hasattr(model, 'model') and hasattr(model.model, 'load_device'):
            model.model.load_device = lead_device
        
        print(f"[ParallelAnything] Parallel setup complete. Devices: {device_names}")
        if has_lora_patches:
            print("[ParallelAnything] LoRA weights synchronized across all devices")
        
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
