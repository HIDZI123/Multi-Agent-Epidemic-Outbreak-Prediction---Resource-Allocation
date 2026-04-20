import pandas as pd
import streamlit as st

from core.analytics import PredictiveAnalytics
from core.knowledge import EnhancedKnowledgeBase
from core.simulation import EpidemicModel
from orchestrators.langchain_orchestrator import LangChainOrchestrator
from orchestrators.langgraph_coordinator import LangGraphCoordinator
from ui.visualizations import epidemic_figure


st.set_page_config(page_title="Epidemic Multi-Agent GUI", layout="wide")
st.title("Notebook 8 + 9 Visual Explainer: LangChain vs LangGraph")
st.caption(
    "This dashboard reproduces the working style of Notebook 08 (LangChain orchestration) and Notebook 09 (LangGraph workflow) without modifying the original notebooks."
)

with st.sidebar:
    st.header("Simulation Controls")
    mode = st.selectbox("Execution Mode", ["LangChain", "LangGraph", "Both"])
    population = st.slider("Population", 200, 1200, 700, 50)
    transmission_rate = st.slider("Base Transmission Rate", 0.1, 1.0, 0.9, 0.05)
    initial_infected = st.slider("Initial Infected Probability", 0.01, 0.6, 0.3, 0.01)
    hospital_capacity = st.slider("Hospital Capacity", 4, 30, 14, 1)
    steps = st.slider("Simulation Steps", 20, 200, 100, 5)
    decision_interval_chain = st.slider("LangChain Decision Interval", 1, 10, 5, 1)
    decision_interval_graph = st.slider("LangGraph Decision Interval", 1, 5, 1, 1)


def run_chain(params):
    model_params = {k: v for k, v in params.items() if k not in ["steps", "chain_interval", "graph_interval", "mode"]}
    model = EpidemicModel(**model_params)
    kb = EnhancedKnowledgeBase()
    analytics = PredictiveAnalytics()
    orchestrator = LangChainOrchestrator(model, kb, analytics)

    step_summaries = []
    for step in range(params["steps"]):
        if not model.running:
            break
        model.step()
        if step % params["chain_interval"] == 0:
            summary = orchestrator.evaluate_and_act(step, decision_threshold=0.35)
            step_summaries.append(summary)

    results_df = model.datacollector.get_model_vars_dataframe()
    decision_df = pd.DataFrame([d.__dict__ for d in orchestrator.decision_log]) if orchestrator.decision_log else pd.DataFrame()
    return results_df, decision_df, step_summaries


def run_graph(params):
    model_params = {k: v for k, v in params.items() if k not in ["steps", "chain_interval", "graph_interval", "mode"]}
    model = EpidemicModel(**model_params)
    kb = EnhancedKnowledgeBase()
    analytics = PredictiveAnalytics()
    coordinator = LangGraphCoordinator(model, kb, analytics, decision_threshold=0.35)

    outputs = []
    for step in range(params["steps"]):
        if not model.running:
            break
        model.step()
        if step % params["graph_interval"] == 0:
            out = coordinator.run_graph_step(step)
            outputs.append(out)

    results_df = model.datacollector.get_model_vars_dataframe()
    trace_df = pd.DataFrame(coordinator.history) if coordinator.history else pd.DataFrame()
    return results_df, trace_df, outputs


if st.button("Run Simulation"):
    params = {
        "N": population,
        "width": 35,
        "height": 35,
        "transmission_rate": transmission_rate,
        "recovery_rate": 0.04,
        "hospitalized_recovery_rate": 0.05,
        "p_initial_infected": initial_infected,
        "num_hospitals": 3,
        "hospital_capacity": hospital_capacity,
        "steps": steps,
        "chain_interval": decision_interval_chain,
        "graph_interval": decision_interval_graph,
    }

    if mode in ["LangChain", "Both"]:
        with st.spinner("Running LangChain-style workflow..."):
            chain_results_df, decision_df, step_summaries = run_chain(params)

        st.subheader("LangChain Flow (Notebook 08 style)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Active Policies", int(chain_results_df["Active Policies"].iloc[-1]))
        c2.metric("Final Hospital Occupancy", int(chain_results_df["Total Hospital Occupancy"].iloc[-1]))
        c3.metric("Final Infected (Untreated)", int(chain_results_df["Infected (Untreated)"].iloc[-1]))

        st.pyplot(epidemic_figure(chain_results_df))
        st.markdown("### Decision Log")
        if decision_df.empty:
            st.info("No policy activations were triggered in this run.")
        else:
            st.dataframe(decision_df.tail(30), width='stretch')

        csv_chain = chain_results_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Download LangChain Results CSV",
            data=csv_chain,
            file_name="langchain_results.csv",
            mime="text/csv",
        )

    if mode in ["LangGraph", "Both"]:
        with st.spinner("Running LangGraph-style workflow..."):
            graph_results_df, trace_df, graph_outputs = run_graph(params)

        st.subheader("LangGraph Flow (Notebook 09 style)")
        g1, g2, g3 = st.columns(3)
        g1.metric("Final Active Policies", int(graph_results_df["Active Policies"].iloc[-1]))
        g2.metric("Final Hospital Occupancy", int(graph_results_df["Total Hospital Occupancy"].iloc[-1]))
        g3.metric("Final Infected (Untreated)", int(graph_results_df["Infected (Untreated)"].iloc[-1]))

        st.pyplot(epidemic_figure(graph_results_df))
        st.markdown("### Graph Node Trace")
        if trace_df.empty:
            st.info("No graph traces were recorded in this run.")
        else:
            st.dataframe(trace_df.tail(30), width='stretch')

        csv_graph = graph_results_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Download LangGraph Results CSV",
            data=csv_graph,
            file_name="langgraph_results.csv",
            mime="text/csv",
        )

    if mode == "Both":
        st.subheader("Side-by-Side Comparison")
        left, right = st.columns(2)

        left.markdown("#### LangChain")
        left.line_chart(chain_results_df[["Infected (Untreated)", "Hospitalized", "Active Policies"]])

        right.markdown("#### LangGraph")
        right.line_chart(graph_results_df[["Infected (Untreated)", "Hospitalized", "Active Policies"]])

        comparison = pd.DataFrame(
            {
                "Metric": ["Final Infected (Untreated)", "Final Hospitalized", "Final Active Policies"],
                "LangChain": [
                    int(chain_results_df["Infected (Untreated)"].iloc[-1]),
                    int(chain_results_df["Hospitalized"].iloc[-1]),
                    int(chain_results_df["Active Policies"].iloc[-1]),
                ],
                "LangGraph": [
                    int(graph_results_df["Infected (Untreated)"].iloc[-1]),
                    int(graph_results_df["Hospitalized"].iloc[-1]),
                    int(graph_results_df["Active Policies"].iloc[-1]),
                ],
            }
        )
        st.dataframe(comparison, width='stretch')

st.markdown("---")
st.markdown("### How this maps to Notebooks 08 and 09")
st.markdown(
    "- LangChain panel mirrors Notebook 08 agent-style decision cycles and decision log behavior.\n"
    "- LangGraph panel mirrors Notebook 09 explicit node routing with traceability.\n"
)
