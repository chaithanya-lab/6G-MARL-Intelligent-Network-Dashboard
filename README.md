# 6G MARL Network Resource Optimization

## Project Overview

This project implements a Multi-Agent Reinforcement Learning (MARL)
framework for resource optimization in a simulated 6G network.

The system models multiple network agents that make resource allocation
decisions based on network observations.

## Main Components

- 6G network simulation environment
- Multi-Agent Reinforcement Learning
- PPO-based actor-critic agents
- Monte Carlo uncertainty estimation
- Feature importance / explainability analysis
- MARL evaluation
- Random baseline comparison
- Statistical analysis
- Performance visualization

## Project Structure

```text
6G_MARL_Project/
│
├── agents/
│   └── marl_agent.py
│
├── environment/
│   └── network_env.py
│
├── uncertainty/
│   ├── uncertainty.py
│   └── test_uncertainty.py
│
├── explainability/
│   └── explain.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── baseline.py
│   ├── compare_results.py
│   ├── comparison_plot.py
│   └── statistical_analysis.py
│
├── models/
│
├── results/
│
├── main.py
└── generate_report.py# 6G-MARL-Intelligent-Network-Dashboard
