import torch
import torch.nn as nn
import types
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import comfy.model_management

class ParallelDevice:
    """Individual device configuration node with auto-detected hardware."""
    
    @classmethod
    def get_available_devices(cls):
        """Dynamically detect all available compute devices."""
        devices = ["cpu"]  # CPU always available
        
        # NVIDIA CUDA (also covers AMD ROCm, which uses CUDA interface in PyTorch)
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                # Format: "cuda:0 (NVIDIA RTX 4090)" truncated for readability
                safe_name = name[:25] + "..." if len(name) > 28 else name
                devices.append(f"cuda:{i}")
        
        # Apple Silicon (Metal Performance Shaders)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
        
        # Intel XPU (Arc GPUs, etc.)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            count = torch.xpu.device_count()
            for i in range(count):
                devices.append(f"xpu:{i}")
        
        # DirectML (Windows ML devices)
        try:
            import torch_directml
            dml_count = torch_directml.device_count()
            for i in range(dml_count):
                devices.append(f"privateuseone:{i}")  # DirectML device naming
        except ImportError:
            pass
            
        return devices

    @classmethod
    def INPUT_TYPES(s):
        available = s.get_available_devices()
        # Prefer cuda:0 as default if available
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
            
        # device_id comes directly from dropdown selection
        config = {
            "device": device_id,
            "percentage": float(percentage),
            "weight": float(percentage) / 100.0
        }
        
        # Append to chain
        new_chain = previous_devices.copy()
        new_chain.append(config)
        
        return (new_chain,)

class ParallelDeviceList:
    """Alternative: Parallel layout with dropdowns for 1-4 devices."""
    
    @classmethod
    def get_available_devices(cls):
        """Get devices for dropdown options."""
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
                "workload_split": ("BOOLEAN", {"default": True, "tooltip": "Enable multi-device processing"}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "setup_parallel"
    CATEGORY = "utils/hardware"

    def setup_parallel(self, model, device_chain, workload_split=True):
        if model is None or not device_chain:
            return (model,)
            
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
        target_model = model.model.diffusion_model
        
        if hasattr(target_model, "_true_parallel_active"):
            return (model,)
            
        # Validate devices
        for dev in device_names:
            try:
                torch.device(dev)
            except:
                print(f"[ParallelAnything] Invalid device: {dev}")
                return (model,)
        
        # Create replicas (N× VRAM)
        replicas = {}
        try:
            print(f"[ParallelAnything] Cloning to {len(device_names)} devices...")
            state_dict = copy.deepcopy(target_model.state_dict())
            
            for dev_name in device_names:
                dev = torch.device(dev_name)
                
                try:
                    model_class = target_model.__class__
                    if hasattr(target_model, 'config'):
                        replica = model_class(**target_model.config)
                    elif hasattr(target_model, 'unet_config'):
                        replica = model_class(**target_model.unet_config)
                    else:
                        replica = model_class()
                    replica.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    replica = copy.deepcopy(target_model)
                    
                replica.to(dev).eval()
                for param in replica.parameters():
                    param.requires_grad = False
                replicas[dev_name] = replica
                print(f"[ParallelAnything] ✓ {dev_name}")
                
        except RuntimeError as e:
            print(f"[ParallelAnything] VRAM Error: {e}")
            return (model,)
            
        # Store references for closure
        replicas_ref = replicas
        devices_ref = device_names
        weights_ref = weights
        lead_ref = lead_device

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
                        'size': size
                    })
            
            if len(active) == 1:
                with torch.no_grad():
                    return active[0]['replica'](x, timesteps, context=context, **kwargs)
            
            # Split tensors
            x_chunks = torch.split(x, [a['size'] for a in active], dim=0)
            t_chunks = torch.split(timesteps, [a['size'] for a in active], dim=0)
            c_chunks = torch.split(context, [a['size'] for a in active], dim=0) if context is not None else [None] * len(active)
            
            results = [None] * len(active)
            
            def worker(task_idx):
                task = active[task_idx]
                dev = task['device']
                replica = task['replica']
                
                try:
                    # Handle device context
                    if dev.type == 'cuda' or dev.type == 'xpu':
                        with torch.cuda.device(dev) if dev.type == 'cuda' else torch.xpu.device(dev):
                            torch.cuda.synchronize(dev) if dev.type == 'cuda' else torch.xpu.synchronize(dev)
                            
                            x_in = x_chunks[task_idx].to(dev, non_blocking=True)
                            t_in = t_chunks[task_idx].to(dev, non_blocking=True)
                            c_in = c_chunks[task_idx].to(dev, non_blocking=True) if c_chunks[task_idx] is not None else None
                            
                            with torch.no_grad():
                                out = replica(x_in, t_in, context=c_in, **kwargs)
                                
                            torch.cuda.synchronize(dev) if dev.type == 'cuda' else torch.xpu.synchronize(dev)
                            return task_idx, out.to(lead_ref, non_blocking=False)
                    else:
                        # CPU/MPS don't need explicit context managers
                        x_in = x_chunks[task_idx].to(dev)
                        t_in = t_chunks[task_idx].to(dev)
                        c_in = c_chunks[task_idx].to(dev) if c_chunks[task_idx] is not None else None
                        
                        with torch.no_grad():
                            out = replica(x_in, t_in, context=c_in, **kwargs)
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
        
        target_model._original_forward = target_model.forward
        target_model.forward = types.MethodType(parallel_forward, target_model)
        target_model._true_parallel_active = True
        target_model._parallel_replicas = replicas
        model.load_device = lead_device
        
        return (model,)

NODE_CLASS_MAPPINGS = {
    "ParallelAnything": ParallelAnything,
    "ParallelDevice": ParallelDevice,
    "ParallelDeviceList": ParallelDeviceList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAnything": "Parallel Anything (True Multi-GPU)", 
    "ParallelDevice": "Parallel Device Config",
    "ParallelDeviceList": "Parallel Device List (1-4x)"
}
