import pandas as pd
import numpy as np
import os


def confidence_interval(series):
    mean = series.mean()
    std = series.std()
    n = len(series)

    margin = 1.96 * (std / np.sqrt(n))

    return mean, std, mean - margin, mean + margin


def main():

    marl_file = "results/evaluation_results.csv"
    random_file = "results/random_baseline_results.csv"

    marl = pd.read_csv(marl_file)
    random = pd.read_csv(random_file)

    metrics = [
        "reward",
        "throughput",
        "latency",
        "pdr",
        "energy"
    ]

    rows = []

    for metric in metrics:

        marl_mean, marl_std, marl_low, marl_high = \
            confidence_interval(marl[metric])

        random_mean, random_std, random_low, random_high = \
            confidence_interval(random[metric])

        rows.append({
            "Metric": metric,
            "MARL Mean": marl_mean,
            "MARL Std": marl_std,
            "MARL CI Lower": marl_low,
            "MARL CI Upper": marl_high,
            "Random Mean": random_mean,
            "Random Std": random_std,
            "Random CI Lower": random_low,
            "Random CI Upper": random_high
        })

    results = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)

    results.to_csv(
        "results/statistical_analysis.csv",
        index=False
    )

    print("\n==========================================")
    print("STATISTICAL ANALYSIS")
    print("==========================================\n")

    print(results.round(4).to_string(index=False))

    print("\nSaved to:")
    print("results/statistical_analysis.csv")


if __name__ == "__main__":
    main()