import os
import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILE = "results/evaluation_results.csv"
OUTPUT_DIR = "results"


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Load evaluation results
    data = pd.read_csv(
        RESULT_FILE
    )

    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")

    print(
        data.describe()
    )

    # -----------------------------
    # Reward
    # -----------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        data["reward"]
    )

    plt.xlabel(
        "Evaluation Episode"
    )

    plt.ylabel(
        "Total Reward"
    )

    plt.title(
        "MARL Reward Across Evaluation Episodes"
    )

    plt.tight_layout()

    plt.savefig(
        "results/reward_curve.png",
        dpi=300
    )

    plt.close()

    # -----------------------------
    # Throughput
    # -----------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        data["throughput"]
    )

    plt.xlabel(
        "Evaluation Episode"
    )

    plt.ylabel(
        "Throughput"
    )

    plt.title(
        "Network Throughput"
    )

    plt.tight_layout()

    plt.savefig(
        "results/throughput_curve.png",
        dpi=300
    )

    plt.close()

    # -----------------------------
    # Latency
    # -----------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        data["latency"]
    )

    plt.xlabel(
        "Evaluation Episode"
    )

    plt.ylabel(
        "Latency (ms)"
    )

    plt.title(
        "Network Latency"
    )

    plt.tight_layout()

    plt.savefig(
        "results/latency_curve.png",
        dpi=300
    )

    plt.close()

    # -----------------------------
    # PDR
    # -----------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        data["pdr"]
    )

    plt.xlabel(
        "Evaluation Episode"
    )

    plt.ylabel(
        "Packet Delivery Ratio"
    )

    plt.title(
        "Packet Delivery Ratio"
    )

    plt.tight_layout()

    plt.savefig(
        "results/pdr_curve.png",
        dpi=300
    )

    plt.close()

    # -----------------------------
    # Energy
    # -----------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        data["energy"]
    )

    plt.xlabel(
        "Evaluation Episode"
    )

    plt.ylabel(
        "Energy Consumption"
    )

    plt.title(
        "Energy Consumption"
    )

    plt.tight_layout()

    plt.savefig(
        "results/energy_curve.png",
        dpi=300
    )

    plt.close()

    print("\nGraphs generated successfully!")

    print("\nSaved files:")

    print(
        "results/reward_curve.png"
    )

    print(
        "results/throughput_curve.png"
    )

    print(
        "results/latency_curve.png"
    )

    print(
        "results/pdr_curve.png"
    )

    print(
        "results/energy_curve.png"
    )


if __name__ == "__main__":
    main()