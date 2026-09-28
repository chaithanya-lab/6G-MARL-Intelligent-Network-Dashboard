import pandas as pd
import matplotlib.pyplot as plt
import os


def main():

    file = "results/marl_vs_random.csv"

    data = pd.read_csv(file, index_col=0)

    os.makedirs("results", exist_ok=True)

    metrics = [
        "reward",
        "throughput",
        "latency",
        "pdr",
        "energy"
    ]

    for metric in metrics:

        values = data.loc[metric]

        plt.figure(figsize=(7, 5))

        values.plot(kind="bar")

        plt.title(f"MARL vs Random Baseline - {metric.title()}")
        plt.ylabel(metric.title())
        plt.xlabel("Method")

        plt.xticks(rotation=0)

        plt.tight_layout()

        filename = f"results/{metric}_marl_vs_random.png"

        plt.savefig(filename, dpi=300)

        plt.close()

        print("Saved:", filename)

    print("\n================================")
    print("COMPARISON PLOTS COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()
    