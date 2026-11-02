# # cat > benchmark.py <<'PY'
# import time
# import torch

# N = 2048
# WARMUP = 5
# ITERATIONS = 20

# print("PyTorch:", torch.__version__)
# print("GPU:", torch.cuda.get_device_name(0))

# def benchmark(device):
#     # Same workload on each device
#     a = torch.randn(N, N, device=device)
#     b = torch.randn(N, N, device=device)

#     for _ in range(WARMUP):
#         c = torch.matmul(a, b)

#     if device == "cuda":
#         torch.cuda.synchronize()

#     start = time.perf_counter()

#     for _ in range(ITERATIONS):
#         c = torch.matmul(a, b)

#     if device == "cuda":
#         torch.cuda.synchronize()

#     elapsed = time.perf_counter() - start
#     return elapsed, elapsed / ITERATIONS

# # Run CPU first, then GPU
# cpu_total, cpu_avg = benchmark("cpu")
# gpu_total, gpu_avg = benchmark("cuda")

# print("\n===== RESULTS =====")
# print(f"Matrix size: {N} x {N}")
# print(f"Iterations: {ITERATIONS}")
# print(f"CPU total: {cpu_total:.3f} s")
# print(f"GPU total: {gpu_total:.3f} s")
# print(f"CPU average: {cpu_avg:.4f} s/iteration")
# print(f"GPU average: {gpu_avg:.4f} s/iteration")
# print(f"GPU speedup: {cpu_total / gpu_total:.2f}x")
# # PY

# cat > src/nn_benchmark.py <<'PY'
import time
import torch
import torch.nn as nn

torch.manual_seed(42)

DEVICE_GPU = "cuda"
BATCH_SIZE = 128
INPUT_SIZE = 512
HIDDEN_SIZE = 256
NUM_CLASSES = 10
WARMUP = 10
ITERATIONS = 100

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

class SmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

def benchmark(device):
    model = SmallNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(BATCH_SIZE, INPUT_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    for _ in range(WARMUP):
        train_step()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        train_step()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed

cpu_time = benchmark("cpu")
gpu_time = benchmark("cuda")

print("\n===== NEURAL NETWORK TRAINING RESULTS =====")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training iterations: {ITERATIONS}")
print(f"CPU total time: {cpu_time:.4f} seconds")
print(f"GPU total time: {gpu_time:.4f} seconds")
print(f"CPU time per iteration: {cpu_time/ITERATIONS*1000:.3f} ms")
print(f"GPU time per iteration: {gpu_time/ITERATIONS*1000:.3f} ms")
print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
# PY

# python src/nn_benchmark.py