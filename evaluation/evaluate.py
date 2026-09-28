import os
import numpy as np

from environment.network_env import SixGNetworkEnv
from agents.marl_agent import PPOAgent


MODEL_DIRECTORY = "models"
RESULT_DIRECTORY = "results"

EPISODES = 20


def evaluate():

    os.makedirs(
        RESULT_DIRECTORY,
        exist_ok=True
    )

    env = SixGNetworkEnv(
        n_agents=3,
        max_cycles=100
    )

    agents = {}

    for agent_name in env.possible_agents:

        agent = PPOAgent(
            obs_dim=6,
            action_dim=3
        )

        model_path = os.path.join(
            MODEL_DIRECTORY,
            f"{agent_name}.pt"
        )

        agent.load(model_path)

        agents[agent_name] = agent

    all_rewards = []
    all_throughput = []
    all_latency = []
    all_pdr = []
    all_energy = []

    print("\n==============================")
    print("6G MARL MODEL EVALUATION")
    print("==============================")

    for episode in range(1, EPISODES + 1):

        observations, _ = env.reset(
            seed=1000 + episode
        )

        episode_reward = 0.0
        episode_throughput = []
        episode_latency = []
        episode_pdr = []
        episode_energy = []

        for step in range(env.max_cycles):

            actions = {}

            for agent_name in env.agents:

                action, _, _ = agents[
                    agent_name
                ].act(
                    observations[agent_name],
                    deterministic=True
                )

                actions[agent_name] = action

            (
                observations,
                rewards,
                terminated,
                truncated,
                infos
            ) = env.step(actions)

            for agent_name in env.agents:

                episode_reward += rewards[
                    agent_name
                ]

                episode_throughput.append(
                    infos[agent_name][
                        "throughput"
                    ]
                )

                episode_latency.append(
                    infos[agent_name][
                        "latency"
                    ]
                )

                episode_pdr.append(
                    infos[agent_name][
                        "pdr"
                    ]
                )

                episode_energy.append(
                    infos[agent_name][
                        "energy"
                    ]
                )

            if all(truncated.values()):
                break

        all_rewards.append(
            episode_reward
        )

        all_throughput.append(
            np.mean(episode_throughput)
        )

        all_latency.append(
            np.mean(episode_latency)
        )

        all_pdr.append(
            np.mean(episode_pdr)
        )

        all_energy.append(
            np.mean(episode_energy)
        )

        print(
            f"Episode {episode:02d} | "
            f"Reward: {episode_reward:.3f} | "
            f"Throughput: {all_throughput[-1]:.3f} | "
            f"Latency: {all_latency[-1]:.2f} ms | "
            f"PDR: {all_pdr[-1]:.3f}"
        )

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(
        f"Average Reward: "
        f"{np.mean(all_rewards):.4f}"
    )

    print(
        f"Average Throughput: "
        f"{np.mean(all_throughput):.4f}"
    )

    print(
        f"Average Latency: "
        f"{np.mean(all_latency):.4f} ms"
    )

    print(
        f"Average PDR: "
        f"{np.mean(all_pdr):.4f}"
    )

    print(
        f"Average Energy: "
        f"{np.mean(all_energy):.4f}"
    )

    results = np.column_stack(
        (
            all_rewards,
            all_throughput,
            all_latency,
            all_pdr,
            all_energy
        )
    )

    results_path = os.path.join(
        RESULT_DIRECTORY,
        "evaluation_results.csv"
    )

    np.savetxt(
        results_path,
        results,
        delimiter=",",
        header="reward,throughput,latency,pdr,energy",
        comments=""
    )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    evaluate()