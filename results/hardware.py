import platform
import psutil
import GPUtil
import os

def get_cpu_info():
    print("=== CPU Info ===")
    print(f"Processor      : {platform.processor()}")
    print(f"CPU Cores      : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical")
    print(f"CPU Frequency  : {psutil.cpu_freq().current:.2f} MHz")

def get_gpu_info():
    print("\n=== GPU Info ===")
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPU found.")
    else:
        for gpu in gpus:
            print(f"Name           : {gpu.name}")
            print(f"Driver Version : {gpu.driver}")
            print(f"Memory Total   : {gpu.memoryTotal}MB")
            print(f"Memory Used    : {gpu.memoryUsed}MB")
            print(f"Memory Free    : {gpu.memoryFree}MB")
            print(f"GPU Load       : {gpu.load * 100:.1f}%")
            print(f"Temperature    : {gpu.temperature} °C\n")

def get_system_info():
    print("=== System Info ===")
    print(f"OS             : {platform.system()} {platform.release()}")
    print(f"Machine        : {platform.machine()}")
    print(f"RAM            : {psutil.virtual_memory().total / 1e9:.2f} GB")

if __name__ == "__main__":
    get_system_info()
    get_cpu_info()
    get_gpu_info()
