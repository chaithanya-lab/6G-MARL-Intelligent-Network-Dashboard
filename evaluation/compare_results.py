import pandas as pd
import os


def main():

    marl_file = "results/evaluation_results.csv"
    random_file = "results/random_baseline_results.csv"

    marl = pd.read_csv(marl_file)
    random = pd.read_csv(random_file)

    marl_avg = marl.mean(numeric_only=True)
    random_avg = random.mean(numeric_only=True)

    comparison = pd.DataFrame({
        "MARL": marl_avg,
        "Random Baseline": random_avg
    })

    comparison = comparison.loc[
        ["reward", "throughput", "latency", "pdr", "energy"]
    ]

    print("\n========================================")
    print("MARL vs RANDOM BASELINE")
    print("========================================\n")

    print(comparison.round(4))

    os.makedirs("results", exist_ok=True)

    comparison.to_csv(
        "results/marl_vs_random.csv"
    )

    print("\nComparison saved to:")
    print("results/marl_vs_random.csv")


if __name__ == "__main__":
    main()