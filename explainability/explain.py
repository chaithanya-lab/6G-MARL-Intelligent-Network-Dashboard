import os
import numpy as np
import matplotlib.pyplot as plt

from environment.network_env import SixGNetworkEnv
from agents.marl_agent import PPOAgent


FEATURE_NAMES = [
    "Traffic",
    "Bandwidth",
    "Channel Quality",
    "Latency",
    "Energy",
    "User Load"
]


def calculate_feature_importance(
    agent,
    observation
):
    """
    Estimate feature importance by
    changing each input feature and
    measuring the change in action probabilities.
    """

    observation = np.asarray(
        observation,
        dtype=np.float32
    )

    # Original prediction
    original_logits, _ = agent.model(
        __import__("torch").tensor(
            observation,
            dtype=__import__("torch").float32
        ).unsqueeze(0)
    )

    torch = __import__("torch")

    original_probabilities = torch.softmax(
        original_logits,
        dim=-1
    ).detach().numpy()[0]

    importance = []

    for feature_index in range(
        len(observation)
    ):

        modified = observation.copy()

        # Perturb the feature
        modified[feature_index] = np.clip(
            modified[feature_index] + 0.1,
            0.0,
            1.0
        )

        modified_tensor = torch.tensor(
            modified,
            dtype=torch.float32
        ).unsqueeze(0)

        modified_logits, _ = agent.model(
            modified_tensor
        )

        modified_probabilities = torch.softmax(
            modified_logits,
            dim=-1
        ).detach().numpy()[0]

        difference = np.mean(
            np.abs(
                modified_probabilities
                - original_probabilities
            )
        )

        importance.append(
            float(difference)
        )

    return np.array(
        importance
    )


def main():

    os.makedirs(
        "results",
        exist_ok=True
    )

    env = SixGNetworkEnv(
        n_agents=3,
        max_cycles=10
    )

    observations, _ = env.reset(
        seed=42
    )

    print("\n==============================")
    print("MODEL EXPLAINABILITY")
    print("==============================")

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

        importance = calculate_feature_importance(
            agent,
            observations[agent_name]
        )

        print(
            f"\n{agent_name} feature importance:"
        )

        for name, value in zip(
            FEATURE_NAMES,
            importance
        ):

            print(
                f"{name}: {value:.6f}"
            )

        # Save graph
        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            FEATURE_NAMES,
            importance
        )

        plt.xlabel(
            "Network Feature"
        )

        plt.ylabel(
            "Importance"
        )

        plt.title(
            f"Feature Importance - {agent_name}"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.tight_layout()

        graph_path = os.path.join(
            "results",
            f"{agent_name}_feature_importance.png"
        )

        plt.savefig(
            graph_path,
            dpi=300
        )

        plt.close()

        print(
            f"Graph saved: {graph_path}"
        )

    print("\n==============================")
    print("Explainability analysis complete!")
    print("==============================")


if __name__ == "__main__":
    main()