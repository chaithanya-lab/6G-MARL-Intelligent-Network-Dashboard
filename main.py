from environment.network_env import SixGNetworkEnv


env = SixGNetworkEnv(
    n_agents=3,
    max_cycles=5,
    seed=42
)

observations, _ = env.reset()

print("\n==============================")
print("6G NETWORK ENVIRONMENT TEST")
print("==============================")

print("\nInitial Network State:")

for agent, state in observations.items():

    print(
        agent,
        "->",
        state
    )


for step in range(5):

    actions = {
        agent:
        env.action_spaces[agent].sample()
        for agent in env.agents
    }

    (
        observations,
        rewards,
        terminated,
        truncated,
        infos
    ) = env.step(actions)

    print(
        f"\nStep {step + 1}"
    )

    print(
        "Actions:",
        actions
    )

    print(
        "Rewards:",
        rewards
    )

    for agent in infos:

        print(
            agent,
            "| Throughput:",
            round(
                infos[agent]["throughput"],
                3
            ),
            "| Latency:",
            round(
                infos[agent]["latency"],
                2
            ),
            "ms",
            "| PDR:",
            round(
                infos[agent]["pdr"],
                3
            )
        )


print(
    "\n=============================="
)

print(
    "Environment test completed!"
)

print(
    "=============================="
)