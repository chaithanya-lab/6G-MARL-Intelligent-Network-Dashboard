import numpy as np
import torch


def monte_carlo_uncertainty(
    agent,
    observation,
    n_samples=50
):
    """
    Estimate action uncertainty using
    Monte Carlo Dropout.

    Returns:
        mean_probabilities
        uncertainty
        predicted_action
    """

    # Enable dropout during inference
    agent.model.train()

    observation = torch.tensor(
        observation,
        dtype=torch.float32
    ).unsqueeze(0)

    probabilities = []

    with torch.no_grad():

        for _ in range(n_samples):

            logits, _ = agent.model(
                observation
            )

            probs = torch.softmax(
                logits,
                dim=-1
            )

            probabilities.append(
                probs.squeeze(0).numpy()
            )

    probabilities = np.array(
        probabilities
    )

    mean_probabilities = np.mean(
        probabilities,
        axis=0
    )

    # Variance across stochastic predictions
    uncertainty = np.mean(
        np.var(
            probabilities,
            axis=0
        )
    )

    predicted_action = int(
        np.argmax(
            mean_probabilities
        )
    )

    return (
        mean_probabilities,
        float(uncertainty),
        predicted_action
    )