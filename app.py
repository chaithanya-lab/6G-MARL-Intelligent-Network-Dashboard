import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="6G MARL Network Dashboard",
    page_icon="📡",
    layout="wide"
)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def load_csv(filename):
    path = RESULTS_DIR / filename

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Could not read {filename}: {e}")
            return None

    return None


def show_image(filename, title=None):
    path = RESULTS_DIR / filename

    if path.exists():
        if title:
            st.subheader(title)
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"{filename} not found.")


def find_column(df, names):
    """Find a column ignoring capitalization and spaces."""
    if df is None:
        return None

    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for name in names:
        key = name.lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------
evaluation_df = load_csv("evaluation_results.csv")
comparison_df = load_csv("marl_vs_random.csv")
statistics_df = load_csv("statistical_analysis.csv")

report_path = RESULTS_DIR / "final_results_report.txt"

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("📡 6G MARL Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Evaluation Results",
        "⚖️ MARL vs Random",
        "📈 Performance Graphs",
        "🔬 Statistical Analysis",
        "🧠 Explainability",
        "🎯 Uncertainty",
        "📄 Final Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Multi-Agent Reinforcement Learning "
    "for intelligent 6G network resource allocation."
)

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.title("📡 6G Multi-Agent Reinforcement Learning")
    st.subheader("Intelligent Network Resource Allocation Dashboard")

    st.markdown(
        """
        This dashboard presents the results of a simulated 6G
        multi-agent reinforcement learning system.

        The system uses multiple agents to make resource-allocation
        decisions based on network conditions such as traffic,
        bandwidth, channel quality, latency, energy and user load.
        """
    )

    st.markdown("---")

    # Top information cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🤖 Agents", "3")
    col2.metric("🎯 Actions", "3")
    col3.metric("📊 Evaluation Episodes", "20")
    col4.metric("🧠 Learning", "PPO")

    st.markdown("---")

    st.subheader("System Architecture")

    st.code(
        """
        6G Network Environment
                  ↓
             Network State
                  ↓
        ┌─────────┼─────────┐
        ↓         ↓         ↓
     Agent 0   Agent 1   Agent 2
        ↓         ↓         ↓
        └─────────┼─────────┘
                  ↓
            PPO Decisions
                  ↓
        Resource Allocation
                  ↓
        ┌─────────┼─────────┐
        ↓         ↓         ↓
    Throughput  PDR     Latency
                  ↓
              Energy
                  ↓
          Performance Results
        """
    )

    st.markdown("---")

    st.subheader("Available Project Results")

    result_files = [
        "evaluation_results.csv",
        "random_baseline_results.csv",
        "marl_vs_random.csv",
        "statistical_analysis.csv",
        "final_results_report.txt"
    ]

    for filename in result_files:
        path = RESULTS_DIR / filename

        if path.exists():
            st.success(f"✓ {filename}")
        else:
            st.warning(f"Missing: {filename}")


# =========================================================
# EVALUATION RESULTS
# =========================================================
elif page == "📊 Evaluation Results":

    st.title("📊 MARL Evaluation Results")

    if evaluation_df is not None:

        st.success("Evaluation results loaded from the project.")

        st.dataframe(
            evaluation_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Evaluation Summary")

        numeric_columns = evaluation_df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) > 0:

            summary = evaluation_df[numeric_columns].mean()

            cols = st.columns(min(4, len(summary)))

            for i, (name, value) in enumerate(summary.items()):

                with cols[i % len(cols)]:
                    st.metric(
                        str(name),
                        f"{value:.4f}"
                    )

    else:
        st.error(
            "evaluation_results.csv was not found."
        )


# =========================================================
# MARL VS RANDOM
# =========================================================
elif page == "⚖️ MARL vs Random":

    st.title("⚖️ MARL vs Random Baseline")

    if comparison_df is not None:

        st.success(
            "MARL and random baseline results loaded."
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Comparison Visualization")

        show_image(
            "reward_marl_vs_random.png",
            "Reward"
        )

        show_image(
            "throughput_marl_vs_random.png",
            "Throughput"
        )

        show_image(
            "latency_marl_vs_random.png",
            "Latency"
        )

        show_image(
            "pdr_marl_vs_random.png",
            "Packet Delivery Ratio"
        )

        show_image(
            "energy_marl_vs_random.png",
            "Energy"
        )

    else:
        st.error(
            "marl_vs_random.csv was not found."
        )


# =========================================================
# PERFORMANCE GRAPHS
# =========================================================
elif page == "📈 Performance Graphs":

    st.title("📈 Performance Analysis")

    st.markdown(
        "Performance graphs generated from the experimental results."
    )

    show_image(
        "reward_curve.png",
        "Reward Curve"
    )

    show_image(
        "throughput_curve.png",
        "Throughput"
    )

    show_image(
        "latency_curve.png",
        "Latency"
    )

    show_image(
        "pdr_curve.png",
        "Packet Delivery Ratio"
    )

    show_image(
        "energy_curve.png",
        "Energy"
    )


# =========================================================
# STATISTICAL ANALYSIS
# =========================================================
elif page == "🔬 Statistical Analysis":

    st.title("🔬 Statistical Analysis")

    if statistics_df is not None:

        st.success(
            "Statistical analysis loaded from the project."
        )

        st.dataframe(
            statistics_df,
            use_container_width=True
        )

        st.markdown("---")

        st.info(
            "The statistical analysis summarizes the experimental "
            "results using measures such as mean, standard deviation "
            "and confidence intervals."
        )

    else:
        st.error(
            "statistical_analysis.csv was not found."
        )


# =========================================================
# EXPLAINABILITY
# =========================================================
elif page == "🧠 Explainability":

    st.title("🧠 Agent Explainability")

    st.markdown(
        """
        The explainability module studies how changes in network-state
        features influence the action probabilities of the MARL agents.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            "agent_0_feature_importance.png",
            "Agent 0 Feature Importance"
        )

        show_image(
            "agent_2_feature_importance.png",
            "Agent 2 Feature Importance"
        )

    with col2:
        show_image(
            "agent_1_feature_importance.png",
            "Agent 1 Feature Importance"
        )

    st.markdown("---")

    st.subheader("Network Features")

    features = [
        "Traffic",
        "Bandwidth",
        "Channel Quality",
        "Latency",
        "Energy",
        "User Load"
    ]

    for feature in features:
        st.write(f"• {feature}")


# =========================================================
# UNCERTAINTY
# =========================================================
elif page == "🎯 Uncertainty":

    st.title("🎯 Uncertainty Analysis")

    st.markdown(
        """
        The uncertainty module uses stochastic inference to examine
        the stability of agent decisions.

        Monte-Carlo dropout is used to generate multiple stochastic
        predictions. The variation between these predictions provides
        an uncertainty estimate.
        """
    )

    st.markdown("---")

    st.subheader("Uncertainty Pipeline")

    st.code(
        """
        Network State
             ↓
        Trained Agent
             ↓
        Multiple Stochastic Predictions
             ↓
        Action Probabilities
             ↓
        Prediction Variance
             ↓
        Uncertainty Estimate
        """
    )

    st.info(
        "The uncertainty values are generated by the existing "
        "uncertainty module in the project."
    )


# =========================================================
# FINAL REPORT
# =========================================================
elif page == "📄 Final Report":

    st.title("📄 Final Experimental Report")

    if report_path.exists():

        try:
            report_text = report_path.read_text(
                encoding="utf-8"
            )

            st.text_area(
                "Final Results Report",
                report_text,
                height=600
            )

        except Exception as e:
            st.error(
                f"Could not read final report: {e}"
            )

    else:
        st.error(
            "final_results_report.txt was not found."
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "6G MARL Project | Simulation-based research dashboard"
)
