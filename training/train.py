import os
import numpy as np

from environment.network_env import SixGNetworkEnv
from agents.marl_agent import PPOAgent


EPISODES = 200

MODEL_DIRECTORY = "models"


def calculate_returns(rewards, gamma=0.99):

    returns = np.zeros(
        len(rewards),
        dtype=np.float32
    )

    running_return = 0.0

    for i in range(
        len(rewards) - 1,
        -1,
        -1
    ):

        running_return = (
            rewards[i]
            + gamma * running_return
        )

        returns[i] = running_return

    return returns


def main():

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True
    )

    env = SixGNetworkEnv(
        n_agents=3,
        max_cycles=100
    )

    observations, _ = env.reset()

    agents = {

        agent: PPOAgent(
            obs_dim=6,
            action_dim=3
        )

        for agent in env.possible_agents
    }

    for episode in range(
        1,
        EPISODES + 1
    ):

        observations, _ = env.reset(
            seed=episode
        )

        buffers = {

            agent: {
                "observations": [],
                "actions": [],
                "log_probabilities": [],
                "rewards": []
            }

            for agent in env.possible_agents
        }

        total_reward = 0.0

        for step in range(
            env.max_cycles
        ):

            actions = {}

            for agent in env.agents:

                (
                    action,
                    log_probability,
                    value
                ) = agents[agent].act(
                    observations[agent]
                )

                actions[agent] = action

                buffers[agent][
                    "observations"
                ].append(
                    observations[agent]
                )

                buffers[agent][
                    "actions"
                ].append(
                    action
                )

                buffers[agent][
                    "log_probabilities"
                ].append(
                    log_probability
                )

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                infos
            ) = env.step(actions)

            for agent in env.agents:

                buffers[agent][
                    "rewards"
                ].append(
                    rewards[agent]
                )

                total_reward += rewards[agent]

            observations = next_observations

            if all(
                truncated.values()
            ):
                break

        for agent in env.possible_agents:

            returns = calculate_returns(
                buffers[agent]["rewards"]
            )

            advantages = (
                returns - returns.mean()
            )

            advantages = (
                advantages /
                (advantages.std() + 1e-8)
            )

            agents[agent].update(
                buffers[agent]["observations"],
                buffers[agent]["actions"],
                buffers[agent]["log_probabilities"],
                returns,
                advantages
            )

        if episode % 10 == 0:

            average_reward = (
                total_reward /
                (
                    env.n_agents *
                    env.max_cycles
                )
            )

            print(
                f"Episode {episode:03d} | "
                f"Average reward = "
                f"{average_reward:.4f}"
            )

    for agent in env.possible_agents:

        agents[agent].save(
            os.path.join(
                MODEL_DIRECTORY,
                f"{agent}.pt"
            )
        )

    print("\nTraining complete.")

    print(
        "Models saved in models/"
    )


if __name__ == "__main__":
    main()