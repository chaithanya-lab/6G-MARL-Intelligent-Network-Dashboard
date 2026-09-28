import pandas as pd
import os

os.makedirs("results", exist_ok=True)

evaluation = pd.read_csv("results/evaluation_results.csv")
comparison = pd.read_csv("results/marl_vs_random.csv", index_col=0)
statistics = pd.read_csv("results/statistical_analysis.csv")

with open("results/final_results_report.txt", "w") as f:

    f.write("6G MARL EXPERIMENT RESULTS\n")
    f.write("==========================\n\n")

    f.write("1. MARL EVALUATION\n")
    f.write("------------------\n")
    f.write(
        evaluation.describe().round(4).to_string()
    )

    f.write("\n\n2. MARL VS RANDOM BASELINE\n")
    f.write("--------------------------\n")
    f.write(
        comparison.round(4).to_string()
    )

    f.write("\n\n3. STATISTICAL ANALYSIS\n")
    f.write("----------------------\n")
    f.write(
        statistics.round(4).to_string(index=False)
    )

    f.write("\n\n4. GENERATED FIGURES\n")
    f.write("--------------------\n")
    f.write("reward_curve.png\n")
    f.write("throughput_curve.png\n")
    f.write("latency_curve.png\n")
    f.write("pdr_curve.png\n")
    f.write("energy_curve.png\n")
    f.write("reward_marl_vs_random.png\n")
    f.write("throughput_marl_vs_random.png\n")
    f.write("latency_marl_vs_random.png\n")
    f.write("pdr_marl_vs_random.png\n")
    f.write("energy_marl_vs_random.png\n")
    f.write("agent_0_feature_importance.png\n")
    f.write("agent_1_feature_importance.png\n")
    f.write("agent_2_feature_importance.png\n")

print("================================")
print("FINAL REPORT GENERATED")
print("================================")
print()
print("Saved to:")
print("results/final_results_report.txt")
