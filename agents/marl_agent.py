import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, action_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),

            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Dropout(0.1)
        )

        self.actor = nn.Linear(
            128,
            action_dim
        )

        self.critic = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        hidden = self.network(x)

        action_logits = self.actor(hidden)

        value = self.critic(hidden).squeeze(-1)

        return action_logits, value


class PPOAgent:

    def __init__(
        self,
        obs_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        clip=0.2
    ):

        self.model = ActorCritic(
            obs_dim,
            action_dim
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.gamma = gamma
        self.clip = clip

    def act(
        self,
        observation,
        deterministic=False
    ):

        if deterministic:
            self.model.eval()
        else:
            self.model.train()

        observation = torch.tensor(
            observation,
            dtype=torch.float32
        ).unsqueeze(0)

        logits, value = self.model(
            observation
        )

        distribution = Categorical(
            logits=logits
        )

        if deterministic:
            action = torch.argmax(
                logits,
                dim=-1
            )
        else:
            action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item())
        )

    def update(
        self,
        observations,
        actions,
        old_log_probabilities,
        returns,
        advantages
    ):

        observations = torch.tensor(
            observations,
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        old_log_probabilities = torch.tensor(
            old_log_probabilities,
            dtype=torch.float32
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32
        )

        for _ in range(4):

            logits, values = self.model(
                observations
            )

            distribution = Categorical(
                logits=logits
            )

            log_probabilities = (
                distribution.log_prob(actions)
            )

            ratio = torch.exp(
                log_probabilities
                - old_log_probabilities
            )

            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip,
                1 + self.clip
            )

            actor_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            critic_loss = 0.5 * (
                returns - values
            ).pow(2).mean()

            entropy = (
                distribution.entropy()
                .mean()
            )

            loss = (
                actor_loss
                + critic_loss
                - 0.01 * entropy
            )

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                0.5
            )

            self.optimizer.step()

    def save(self, path):

        torch.save(
            self.model.state_dict(),
            path
        )

    def load(self, path):

        self.model.load_state_dict(
            torch.load(
                path,
                map_location="cpu"
            )
        )