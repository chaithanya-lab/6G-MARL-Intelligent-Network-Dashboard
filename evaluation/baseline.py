import os
import sys
import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.network_env import SixGNetworkEnv


def evaluate_random_policy(n_episodes=20):
    env = SixGNetworkEnv(
        n_agents=3,
        max_cycles=100,
        seed=42
    )

    all_results = []

    for episode in range(n_episodes):

        observations, _ = env.reset()

        episode_rewards = []
        episode_throughput = []
        episode_latency = []
        episode_pdr = []
        episode_energy = []

        for step in range(100):

            # Random actions
            actions = {
                agent: env.action_spaces[agent].sample()
                for agent in env.agents
            }

            observations, rewards, terminated, truncated, infos = env.step(actions)

            episode_rewards.extend(rewards.values())

            for agent in infos:
                episode_throughput.append(
                    infos[agent]["throughput"]
                )

                episode_latency.append(
                    infos[agent]["latency"]
                )

                episode_pdr.append(
                    infos[agent]["pdr"]
                )

                episode_energy.append(
                    infos[agent]["energy"]
                )

            if all(terminated.values()) or all(truncated.values()):
                break

        all_results.append({
            "episode": episode + 1,
            "reward": np.mean(episode_rewards),
            "throughput": np.mean(episode_throughput),
            "latency": np.mean(episode_latency),
            "pdr": np.mean(episode_pdr),
            "energy": np.mean(episode_energy)
        })

    env.close()

    results = pd.DataFrame(all_results)

    os.makedirs("results", exist_ok=True)

    results.to_csv(
        "results/random_baseline_results.csv",
        index=False
    )

    print("\n==============================")
    print("RANDOM BASELINE RESULTS")
    print("==============================")

    print(
        "\nAverage Reward:",
        results["reward"].mean()
    )

    print(
        "Average Throughput:",
        results["throughput"].mean()
    )

    print(
        "Average Latency:",
        results["latency"].mean()
    )

    print(
        "Average PDR:",
        results["pdr"].mean()
    )

    print(
        "Average Energy:",
        results["energy"].mean()
    )

    print("\nResults saved to:")
    print("results/random_baseline_results.csv")


if __name__ == "__main__":
    evaluate_random_policy()