import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SixGNetworkEnv(ParallelEnv):

    metadata = {
        "name": "6g_marl_v0"
    }

    def __init__(self, n_agents=3, max_cycles=100, seed=42):

        self.n_agents = n_agents
        self.max_cycles = max_cycles

        self.possible_agents = [
            f"agent_{i}"
            for i in range(n_agents)
        ]

        self.agents = self.possible_agents[:]

        # Each agent observes:
        # traffic
        # bandwidth
        # channel quality
        # latency
        # energy
        # user load

        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Actions:
        # 0 = Low resource
        # 1 = Medium resource
        # 2 = High resource

        self.action_spaces = {
            agent: spaces.Discrete(3)
            for agent in self.possible_agents
        }

        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]

        self.t = 0

        # Network conditions
        self.traffic = self.rng.uniform(
            0.2,
            0.8,
            self.n_agents
        )

        self.bandwidth = self.rng.uniform(
            0.5,
            1.0,
            self.n_agents
        )

        self.channel = self.rng.uniform(
            0.4,
            1.0,
            self.n_agents
        )

        self.latency = self.rng.uniform(
            0.1,
            0.3,
            self.n_agents
        )

        self.energy = self.rng.uniform(
            0.2,
            0.5,
            self.n_agents
        )

        self.users = self.rng.uniform(
            0.2,
            0.9,
            self.n_agents
        )

        return (
            self._get_observations(),
            {
                agent: {}
                for agent in self.agents
            }
        )

    def _get_observations(self):

        observations = {}

        for i, agent in enumerate(self.agents):

            observations[agent] = np.array(
                [
                    self.traffic[i],
                    self.bandwidth[i],
                    self.channel[i],
                    self.latency[i],
                    self.energy[i],
                    self.users[i]
                ],
                dtype=np.float32
            )

        return observations

    def step(self, actions):

        rewards = {}
        infos = {}

        self.t += 1

        # Resource allocation levels
        resource_levels = np.array(
            [
                0.4,
                0.7,
                1.0
            ]
        )

        for i, agent in enumerate(self.agents):

            action = int(actions[agent])

            resource = resource_levels[action]

            # Available network capacity
            capacity = (
                self.bandwidth[i]
                * self.channel[i]
                * resource
            )

            demand = self.traffic[i]

            # Throughput
            throughput = min(
                demand,
                capacity
            )

            # Latency
            latency = (
                0.05
                + 0.35
                * demand
                / (capacity + 0.05)
            )

            # Packet delivery ratio
            pdr = float(
                np.clip(
                    self.channel[i]
                    * np.exp(-latency),
                    0.0,
                    1.0
                )
            )

            # Energy consumption
            energy = float(
                0.15
                + 0.55 * resource
                + 0.15 * self.traffic[i]
            )

            # Resource utilization
            utilization = float(
                np.clip(
                    throughput
                    / (capacity + 1e-6),
                    0.0,
                    1.0
                )
            )

            # Reward
            reward = (
                2.0 * throughput
                + 1.0 * pdr
                - 1.2 * latency
                - 0.5 * energy
            )

            rewards[agent] = float(reward)

            infos[agent] = {
                "throughput": float(
                    throughput
                ),

                "latency": float(
                    latency * 1000
                ),

                "pdr": pdr,

                "energy": energy,

                "resource_utilization":
                    utilization,

                "action": action
            }

            # Update network state
            self.latency[i] = latency
            self.energy[i] = energy

            self.traffic[i] = np.clip(
                0.75 * self.traffic[i]
                + 0.25 * self.rng.uniform(
                    0.1,
                    1.0
                ),
                0.0,
                1.0
            )

            self.channel[i] = np.clip(
                0.85 * self.channel[i]
                + 0.15 * self.rng.uniform(
                    0.3,
                    1.0
                ),
                0.0,
                1.0
            )

        observations = self._get_observations()

        terminated = {
            agent: False
            for agent in self.agents
        }

        truncated = {
            agent:
            self.t >= self.max_cycles
            for agent in self.agents
        }

        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos
        )