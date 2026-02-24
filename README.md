# Parallel Anything (True Multi-GPU) for ComfyUI

This suite of nodes unlocks high-performance parallel processing in ComfyUI by utilizing **Model Replication**. Unlike standard offloading which moves a single model instance between devices, these nodes create independent replicas of the model on each selected GPU/CPU, allowing for true simultaneous batch processing.

* **Tested on Z_IMAGE, FLUX.1, WAN2.2**

---

## 🚀 Key Features

* **True Parallel Execution**: Simultaneous forward passes on multiple GPUs using thread-safe model replicas
* **Chainable Device Nodes**: Connect multiple `Parallel Device Config` nodes to easily configure 2-8+ GPUs
* **Auto Hardware Detection**: Dropdown menus automatically populated with available CUDA GPUs, CPU, Apple MPS, and Intel XPU
* **Dynamic Load Balancing**: Percentage-based batch splitting (e.g., 70% on RTX 3090, 30% on RTX 3060)
* **Cross-Platform**: Works on Windows, Linux, and macOS (MPS)

---

## 🛠 Nodes Included

### 1. Parallel Anything (True Multi-GPU)
The main orchestration node. It intercepts the diffusion model's forward pass and triggers simultaneous compute kernels across all available replicas.

<img width="346" height="222" alt="image" src="https://github.com/user-attachments/assets/73096237-fbf7-4116-8187-ed83f07bf94c" />


### 2. Parallel Device Config / List
Allows you to build a `DEVICE_CHAIN`. You can chain multiple GPUs together or use the List node to quickly and their respective workload percentages.

- Option 1
<img width="1397" height="249" alt="image" src="https://github.com/user-attachments/assets/b587b2f3-ef41-4623-a783-4bb2a55dcf31" />

- Option 2
<img width="723" height="315" alt="image" src="https://github.com/user-attachments/assets/9773e615-d1a2-43e5-a1d0-2d00367e78ac" />

- Console
<img width="1072" height="117" alt="image" src="https://github.com/user-attachments/assets/e928ebdb-496d-43a3-9d8e-ab79e03f1a63" />

- GPU Plot
<img width="1517" height="556" alt="image" src="https://github.com/user-attachments/assets/7f477702-e655-439a-b919-89fa1099c685" />

- Workflow Picture

<img width="2054" height="867" alt="image" src="https://github.com/user-attachments/assets/52c1da42-07e3-4cda-8545-96b953a91266" />

- Test Case with batch_Size = 21 (Z_Image Turbo) (1024x1024)
  - PC Specs:
    - AMD Ryzen™ 5 3600
    - ASRock B450M Pro4
    - 128.0 GiB
    - GEN3@16x PCIe (V100) + GEN2@4x PCIe (3090))
    - OS: Linux
      
  - Single 3090 (26.00s/it)

    <img width="1532" height="55" alt="image" src="https://github.com/user-attachments/assets/5cbc348d-fb79-4ea8-b8b5-844ccf2fdb5b" />
    
  - V100 + 3090 (12.91s/it)
    
    <img width="1517" height="49" alt="image" src="https://github.com/user-attachments/assets/eaae446b-5128-442b-bfc6-3fbf31d10ec7" />


---

## Requirements

### Hardware
- **Minimum**: 2x GPUs or 1x GPU + CPU (for testing)
- **Recommended**: Identical GPUs for balanced loads
- **VRAM**: Each GPU must independently hold the full model (e.g., SDXL requires ~7GB per GPU)

### Software
- ComfyUI installed and functional
- PyTorch with appropriate backend:

```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch torchvision

# Intel XPU (experimental)
pip install intel-extension-for-pytorch

```
---

## 📦 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/FearL0rd/ComfyUI-ParallelAnything.git
    ```
3.  **Restart ComfyUI.**

---

## Usage

### Basic 2-GPU Setup

```bash
[Parallel Device Config]                           [Parallel Device Config]
      ↓ cuda:0 (50%)                                     ↓ cuda:1 (50%)
      └──────────────→ [Parallel Anything] ←─────────────┘
                              ↓
                        [Your KSampler/etc]
```

1.  Add Parallel Device Config node → Select cuda:0 from dropdown → Set 50%
2.  Add another Parallel Device Config → Connect DEVICE_CHAIN output from first node → Select cuda:1 → Set 50%
3.  Connect final DEVICE_CHAIN to Parallel Anything node
4.  Connect your MODEL from Load Checkpoint → Parallel Anything → KSampler

### Advanced: 4-GPU Load Balancing
Chain 4 devices with different percentages based on GPU memory:

cuda:0 (RTX 3090): 40%
cuda:1 (RTX 3090): 40%
cuda:2 (V100): 15%
cuda:3 (P100): 5%

### Alternative: Parallel Device List

Use Parallel Device List (1-4x) if you prefer a single node with 4 dropdowns instead of chaining.

### CPU + GPU Hybrid

[Parallel Device Config: cpu (20%)] → [Parallel Device Config: cuda:0 (80%)] → [Parallel Anything]

---

## Performance Tips

1. PCIe Bandwidth Matters
Ensure GPUs share the same PCIe switch or CPU root complex:
```bash
# Linux: Check PCIe topology
lspci -tv | grep -i nvidia
```
Avoid configurations where GPUs are on separate NUMA nodes with limited inter-socket bandwidth.

2. Batch Size Optimization
* **Minimum:** Batch size ≥ Number of GPUs
* **Sweet Spot:** Batch size 8-16 for 2-4 GPUs
* **Diminishing Returns:** Very large batches may saturate PCIe transfer bandwidth

3. Identical GPUs Preferred
* **Mixing GPU architectures:** (e.g., RTX 4090 + RTX 3090) works but the faster GPU will wait for the slower one at each step. Use percentage weights to compensate (e.g., 60/40 split).

4. Model Placement
Place Parallel Anything immediately before the KSampler, after all LoRA/weight modifications:

```bash
Load Checkpoint → Load LoRA → [Parallel Anything] → KSampler
```

---

## ⚠️ Important Considerations

* **VRAM Usage**: This node uses **Model Replication**. If you use 2 GPUs, you will use 2 times the VRAM (one copy per card).
* **Batch Size**: Parallelism only triggers if your **Batch Size** is > the number of devices in your chain.
* **Inference Tensors**: The node automatically clones and detaches tensors to bypass PyTorch's "Inference tensors do not track version counter" error common in multi-GPU workflows.

---

## 🔧 Troubleshooting

If you encounter a **RuntimeError** regarding "Inference Tensors":
* Ensure you are using a **Batch Size** large enough to split.
* The node uses a "Deep Detach" strategy (`.detach().clone()`) to satisfy the version counter requirements of the KSampler.
* if you see the message "RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cuda:1 (when checking argument in method wrapper_CUDA_addmm)
" after changing the percentage of the GPU'S for the second run. restart comfyui or add **--cache-none** in the ComfyUI startup arguments

### Slower than single GPU
#### Common causes:

* **PCIe Bottleneck:** Data transfer overhead exceeds compute benefit (common with x4/x8 PCIe slots)
* **Small Batch Size:** Overhead of splitting/merging exceeds parallel benefit. Try batch size ≥ 8.
* **Mixed GPUs:** Fast GPU waiting for slow GPU. Adjust percentages.
* **Apple MPS (Mac) Issues**
MPS backend does not support all operations needed for stable diffusion. If you encounter errors:

* Use CPU exclusively on Mac for stability
* MPS support is experimental

### Thread Safety Errors
If you see **CUDA error: invalid device ordinal** or similar:

* Ensure you're not wrapping the model twice (check for nested Parallel Anything nodes)
* Verify all selected devices exist: torch.device('cuda:2') will fail if you only have 2 GPUs (indices 0 and 1)

## Architecture Details

This node implements Data Parallelism via model replication:

1. Replication: On setup, the model state dict is deep-copied to N devices
2. Batch Split: Input batch is divided by percentage weights
3. Thread Pool: Each chunk is processed in parallel using ThreadPoolExecutor
4. Synchronization: torch.cuda.synchronize() ensures computation completes before returning to lead device
5. Concatenation: Results are gathered and concatenated on the lead device
6. Trade-off: Uses N× VRAM for N× throughput (approximately). Best for multi-GPU workstations with identical GPUs.

## Limitations

* ❌ No model parallelism (splitting layers across GPUs) - Each GPU holds full model
* ❌ No gradient synchronization - Inference only (no training/fine-tuning)
* ❌ Static load balancing - Percentages fixed per run, no dynamic adjustment based on queue depth
* ❌ Memory overhead - Briefly uses 2× model memory per GPU during the load_state_dict phase

### License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


