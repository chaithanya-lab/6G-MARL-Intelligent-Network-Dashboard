import os

from environment.network_env import SixGNetworkEnv
from agents.marl_agent import PPOAgent
from uncertainty.uncertainty import monte_carlo_uncertainty


def main():

    env = SixGNetworkEnv(
        n_agents=3,
        max_cycles=10
    )

    observations, _ = env.reset(
        seed=42
    )

    agents = {}

    for agent_name in env.possible_agents:

        agent = PPOAgent(
            obs_dim=6,
            action_dim=3
        )

        model_path = os.path.join(
            "models",
            f"{agent_name}.pt"
        )

        agent.load(model_path)

        agents[agent_name] = agent

    print("\n==============================")
    print("UNCERTAINTY ESTIMATION")
    print("==============================")

    for agent_name in env.possible_agents:

        (
            probabilities,
            uncertainty,
            action
        ) = monte_carlo_uncertainty(
            agents[agent_name],
            observations[agent_name],
            n_samples=50
        )

        print(f"\n{agent_name}")

        print(
            "Action probabilities:",
            probabilities
        )

        print(
            "Predicted action:",
            action
        )

        print(
            "Uncertainty:",
            round(
                uncertainty,
                6
            )
        )

    print("\n==============================")
    print("Uncertainty test completed!")
    print("==============================")


if __name__ == "__main__":
    main()