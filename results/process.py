import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_FOLDER = "./data"  # Root folder where your *_metrics_*.csv files are
OUTPUT_TABLE = "./processed/research_results.csv"
PLOT_FOLDER = "./processed/plots"
SPEEDUP_CSV = "./processed/speedup_results.csv"

os.makedirs(PLOT_FOLDER, exist_ok=True)

results = []
pattern = r"(cpu|gpu)_metrics_(\d+)_epochs\.csv"

for fname in os.listdir(DATA_FOLDER):
    match = re.match(pattern, fname)
    if match:
        device, epochs = match.groups()
        epochs = int(epochs)
        filepath = os.path.join(DATA_FOLDER, fname)
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"Warning: Empty file {filepath}")
            continue

        # CPU
        if device == "cpu":
            total_time_ms = df['Epoch_time(ms)'].sum()
            avg_time_per_epoch_ms = df['Epoch_time(ms)'].mean()
            time_std_dev_ms = df['Epoch_time(ms)'].std()
            stats = {
                'Epochs': epochs,
                'Device': 'CPU',
                'Final Accuracy': df['Accuracy(%)'].iloc[-1],
                'Avg Epoch Accuracy': df['Accuracy(%)'].mean(),
                'Epoch Accuracy Std': df['Accuracy(%)'].std(),
                'Total Run Time (s)': total_time_ms / 1000,
                'Avg Time/Epoch (s)': avg_time_per_epoch_ms / 1000,
                'Time/Epoch Std (s)': time_std_dev_ms / 1000,
                'Avg Kernel Time/Epoch (s)': '-',
                'Avg Copy Time/Epoch (s)': '-',
                'Compute Ratio': '-'
            }
            results.append(stats)
        # GPU
        elif device == "gpu":
            total_time_ms = df['Epoch_time(ms)'].sum()
            avg_time_per_epoch_ms = df['Epoch_time(ms)'].mean()
            time_std_dev_ms = df['Epoch_time(ms)'].std()
            avg_kernel_time_ms = df['Kernel Time(ms)'].mean()
            avg_copy_time_ms = df['Data Copy Time(ms)'].mean()
            compute_ratio = (avg_kernel_time_ms / avg_time_per_epoch_ms) if avg_time_per_epoch_ms else 0
            stats = {
                'Epochs': epochs,
                'Device': 'GPU',
                'Final Accuracy': df['Accuracy(%)'].iloc[-1],
                'Avg Epoch Accuracy': df['Accuracy(%)'].mean(),
                'Epoch Accuracy Std': df['Accuracy(%)'].std(),
                'Total Run Time (s)': total_time_ms / 1000,
                'Avg Time/Epoch (s)': avg_time_per_epoch_ms / 1000,
                'Time/Epoch Std (s)': time_std_dev_ms / 1000,
                'Avg Kernel Time/Epoch (s)': avg_kernel_time_ms / 1000,
                'Avg Copy Time/Epoch (s)': avg_copy_time_ms / 1000,
                'Compute Ratio': compute_ratio
            }
            results.append(stats)


# Create DataFrame and sort
df = pd.DataFrame(results).sort_values(['Epochs', 'Device'])

# Calculate GPU vs CPU speedup using TOTAL RUN TIME
speedups = []
for epochs in df['Epochs'].unique():
    cpu_row = df.loc[(df['Epochs'] == epochs) & (df['Device'] == 'CPU')]
    gpu_row = df.loc[(df['Epochs'] == epochs) & (df['Device'] == 'GPU')]
    if not cpu_row.empty and not gpu_row.empty:
        cpu_total_time = cpu_row['Total Run Time (s)'].values[0]
        gpu_total_time = gpu_row['Total Run Time (s)'].values[0]
        if gpu_total_time > 0:
            speedup = cpu_total_time / gpu_total_time
            speedups.append({'Epochs': epochs, 'Speedup Factor': speedup})

speedup_df = pd.DataFrame(speedups)
speedup_df.to_csv(SPEEDUP_CSV, index=False)

# Save results table
df.to_csv(OUTPUT_TABLE, index=False)

# Print formatted table for Word
print("\nResearch Results Table (Copy this for Word):")
word_table = df.round(3).fillna('-').astype(str)
print(word_table[['Epochs', 'Device', 'Final Accuracy',
                  'Avg Epoch Accuracy', 'Total Run Time (s)',
                  'Avg Time/Epoch (s)', 'Compute Ratio']]
      .to_string(index=False, justify='center'))

# --- Generate Plots ---

# 1. Final Accuracy vs Total Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Epochs', y='Final Accuracy', hue='Device', marker='o')
plt.title("Final Accuracy vs. Total Epoch Count")
plt.xlabel("Total Epochs in Run")
plt.ylabel("Final Accuracy (%)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f"{PLOT_FOLDER}/final_accuracy_vs_total_epochs.png", dpi=300)
plt.close()

# 2. Average Time per Epoch Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Epochs', y='Avg Time/Epoch (s)', hue='Device')
plt.title("Average Time per Epoch Comparison")
plt.xlabel("Total Epochs in Run")
plt.ylabel("Average Time per Epoch (s)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.savefig(f"{PLOT_FOLDER}/avg_time_per_epoch.png", dpi=300)
plt.close()

# 3. Total Run Time Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Epochs', y='Total Run Time (s)', hue='Device')
plt.title("Total Run Time Comparison")
plt.xlabel("Total Epochs in Run")
plt.ylabel("Total Run Time (s)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.savefig(f"{PLOT_FOLDER}/total_run_time.png", dpi=300)
plt.close()

# 4. Speedup Factor vs Total Epochs
if not speedup_df.empty:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=speedup_df, x='Epochs', y='Speedup Factor', marker='o')
    plt.title("GPU Speedup Factor vs. Total Epoch Count")
    plt.xlabel("Total Epochs in Run")
    plt.ylabel("Speedup Factor (CPU Time / GPU Time)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"{PLOT_FOLDER}/speedup_vs_total_epochs.png", dpi=300)
    plt.close()
else:
    print("Speedup data frame is empty, skipping speedup plot.")

# 5. GPU Time Breakdown (Based on average per-epoch times)
gpu_df = df.loc[df['Device'] == 'GPU'].copy()
if not gpu_df.empty and 'Avg Kernel Time/Epoch (s)' in gpu_df.columns and 'Avg Copy Time/Epoch (s)' in gpu_df.columns:
    plt.figure(figsize=(10, 6))
    gpu_df.set_index('Epochs')[['Avg Kernel Time/Epoch (s)', 'Avg Copy Time/Epoch (s)']].plot.bar(stacked=True)
    plt.title("Average GPU Time Breakdown per Epoch")
    plt.xlabel("Total Epochs in Run")
    plt.ylabel("Average Time per Epoch (s)")
    plt.legend(["Kernel Time", "Data Copy Time"])
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(f"{PLOT_FOLDER}/gpu_avg_time_breakdown.png", dpi=300)
    plt.close()
else:
    print("GPU data frame is empty or missing columns, skipping GPU breakdown plot.")

print("\nProcessing complete. Results saved to:")
print(f"- Research table CSV: {OUTPUT_TABLE}")
print(f"- Speedup CSV: {SPEEDUP_CSV}")
print(f"- Plots: {PLOT_FOLDER}/*.png")
