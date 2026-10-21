import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary results
summary_df = pd.read_csv("results_mistral_ablation/mistral_ablation_summary.csv")

# Set up plot styling for publication
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

loops = summary_df["max_loops"]

# Plot 1: Retrieval & Answer Metrics Progression
ax1.plot(loops, summary_df["MRR@1"], marker='o', linewidth=2, label='MRR@1')
ax1.plot(loops, summary_df["Prec@3"], marker='s', linewidth=2, label='Precision@3')
ax1.plot(loops, summary_df["Answer_F1"], marker='^', linewidth=2, label='Answer F1-Score')
ax1.set_title("Performance Metrics vs. Agent Loop Iterations", fontsize=12, fontweight='bold')
ax1.set_xlabel("Max Agent Loops ($L$)", fontsize=10)
ax1.set_ylabel("Score", fontsize=10)
ax1.set_xticks([1, 2, 3])
ax1.set_ylim(0, 1.1)
ax1.legend(loc='lower right')

# Plot 2: Latency Trade-off
ax2.plot(loops, summary_df["Avg_Latency_ms"] / 1000.0, marker='d', color='crimson', linewidth=2)
ax2.set_title("Execution Latency vs. Agent Loop Iterations", fontsize=12, fontweight='bold')
ax2.set_xlabel("Max Agent Loops ($L$)", fontsize=10)
ax2.set_ylabel("Average Latency per Query (seconds)", fontsize=10)
ax2.set_xticks([1, 2, 3])

plt.tight_layout()
output_path = "results_mistral_ablation/ablation_performance_trends.png"
plt.savefig(output_path, dpi=300)
print(f"Graph successfully saved to {output_path}")
plt.show()