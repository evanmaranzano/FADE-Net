import torch
import time
import numpy as np
import psutil
import torch.backends.cudnn as cudnn
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model import LightweightAgeEstimator
from config import Config

def apply_common_overrides(cfg, args):
    if args.backbone_source is not None:
        cfg.backbone_source = args.backbone_source
    if args.backbone_name is not None:
        cfg.backbone_name = args.backbone_name
    if args.no_pretrained:
        cfg.backbone_pretrained = False


def benchmark_device(device_name, cfg, num_iterations=1000, batch_size=1):
    print(f"\n🚀 Benchmarking on {device_name} (Batch Size: {batch_size})...")
    
    # 1. Setup Model
    device = torch.device(device_name)
    model = LightweightAgeEstimator(cfg).to(device)
    model.eval()
    
    # Enable cuDNN benchmark for GPU
    if device.type == 'cuda':
        cudnn.benchmark = True
        
    # 2. Fake Data
    input_shape = (batch_size, 3, cfg.img_size, cfg.img_size)
    dummy_input = torch.randn(input_shape).to(device)
    
    # 3. Warm-up
    print("  🔥 Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    
    # Synchronize for GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # 4. Measure
    print(f"  ⏱️ Running {num_iterations} iterations...")
    timings = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)
            
    # 5. Stats
    timings = np.array(timings)
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    fps = batch_size / avg_time
    
    print(f"  ✅ Result:")
    print(f"     Latency: {avg_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"     Throughput: {fps:.2f} FPS")
    
    return avg_time, fps


def device_label(device_name):
    device = torch.device(device_name)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpu_name = "CPU"
    try:
        import platform
        cpu_name = platform.processor() or cpu_name
    except Exception:
        pass
    return cpu_name

def main():
    parser = argparse.ArgumentParser(description="FADE-Net speed benchmark")
    parser.add_argument('--iters', type=int, default=None, help='Override iteration count per device')
    parser.add_argument('--batch_size', type=int, default=1, help='Benchmark batch size')
    parser.add_argument('--backbone_source', type=str, choices=['torchvision', 'timm'], help='Backbone provider')
    parser.add_argument('--backbone_name', type=str, help='Backbone model name')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained backbone weights')
    args = parser.parse_args()

    cfg = Config()
    apply_common_overrides(cfg, args)

    print("="*60)
    print("🏁 FADE-Net Age Estimation Speed Benchmark")
    print("="*60)
    
    # CPU info
    print(f"🖥️ CPU Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"🧠 Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected!")
        
    print("-" * 60)
    
    # --- CPU Benchmark ---
    cpu_iters = args.iters if args.iters is not None else 200
    cpu_latency, cpu_fps = benchmark_device('cpu', cfg, num_iterations=cpu_iters, batch_size=args.batch_size)
    
    # --- GPU Benchmark ---
    if torch.cuda.is_available():
        gpu_iters = args.iters if args.iters is not None else 1000
        gpu_latency, gpu_fps = benchmark_device('cuda', cfg, num_iterations=gpu_iters, batch_size=args.batch_size)
    
    # --- Report Summary ---
    print("\n" + "="*60)
    print("📊 Final Report")
    print("="*60)
    print(f"CPU Inference ({device_label('cpu')}): {cpu_fps:.1f} FPS | {cpu_latency*1000:.1f} ms")
    if torch.cuda.is_available():
        print(f"GPU Inference ({device_label('cuda')}): {gpu_fps:.1f} FPS | {gpu_latency*1000:.1f} ms")
        print(f"Speedup Factors: {gpu_fps/cpu_fps:.1f}x faster on GPU")

if __name__ == "__main__":
    main()
