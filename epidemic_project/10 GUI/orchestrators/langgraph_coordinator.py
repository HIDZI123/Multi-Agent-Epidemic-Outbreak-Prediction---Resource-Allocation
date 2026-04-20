import json
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from core.analytics import PredictiveAnalytics
from core.knowledge import EnhancedKnowledgeBase
from core.policy import Policy, PolicyType
from core.simulation import EpidemicModel


class EpidemicGraphState(TypedDict, total=False):
    step: int
    metrics: Dict[str, float]
    predictions: Dict[str, Dict[str, float]]
    candidate_policies: List[str]
    activated_policies: List[str]
    decision_log: List[Dict[str, str]]
    node_trace: List[str]
    interrupt_reason: str


class LangGraphCoordinator:
    def __init__(
        self,
        model: EpidemicModel,
        kb: EnhancedKnowledgeBase,
        analytics: PredictiveAnalytics,
        decision_threshold: float = 0.35,
    ):
        self.model = model
        self.kb = kb
        self.analytics = analytics
        self.decision_threshold = decision_threshold
        self.history: List[Dict[str, object]] = []
        self.graph = self._build_graph()

    def _metrics(self) -> Dict[str, float]:
        return self.model.metrics_snapshot()

    def _predictions(self) -> Dict[str, Dict[str, float]]:
        if len(self.model.datacollector.model_vars) < 15:
            return {}
        df = self.model.datacollector.get_model_vars_dataframe()
        hosp = self.analytics.predict_hospital_demand(df)
        peak = self.analytics.predict_infection_peak(df)
        return {"hospital_demand": hosp.__dict__, "infection_peak": peak.__dict__}

    @staticmethod
    def _policy_from_name(name: str) -> Optional[Policy]:
        mapping = {
            "social_distancing": (PolicyType.SOCIAL_DISTANCING, 0.30, 300),
            "mask_mandate": (PolicyType.MASK_MANDATE, 0.20, 120),
            "elective_surgery_ban": (PolicyType.ELECTIVE_SURGERY_BAN, 0.25, 100),
            "school_closure": (PolicyType.SCHOOL_CLOSURE, 0.20, 350),
            "inter_hospital_transfer": (PolicyType.INTER_HOSPITAL_TRANSFER, 0.10, 80),
            "crisis_standards": (PolicyType.CRISIS_STANDARDS, 0.15, 200),
            "resource_rationing": (PolicyType.RESOURCE_RATIONING, 0.10, 150),
        }
        if name not in mapping:
            return None
        ptype, eff, cost = mapping[name]
        return Policy(ptype, False, eff, cost, 20, f"Graph activation: {name}")

    def _heuristic_policy_selection(self, metrics: Dict[str, float], candidates: List[str]) -> List[str]:
        selected: List[str] = []
        if metrics["hospital_occupancy_rate"] > 0.55:
            selected.extend(["elective_surgery_ban", "inter_hospital_transfer"])
        if metrics["hospital_occupancy_rate"] > 0.75:
            selected.extend(["crisis_standards", "resource_rationing"])
        if metrics["infected_untreated"] > 35 or metrics["effective_transmission_rate"] > 0.20:
            selected.extend(["social_distancing", "mask_mandate"])
        selected.extend(candidates[:3])
        return list(dict.fromkeys(selected))

    def _score_policy(self, metrics: Dict[str, float], policy: Policy) -> float:
        urgency = 0.0
        urgency += 0.5 if metrics["hospital_occupancy_rate"] > 0.60 else 0.0
        urgency += 0.3 if metrics["infected_untreated"] > 40 else 0.0
        urgency += 0.2 if metrics["effective_transmission_rate"] > 0.20 else 0.0
        return min(1.0, urgency * 0.5 + policy.effectiveness * 0.3 + (1 / (1 + policy.implementation_cost / 100)) * 0.2)

    def _activate_policy_set(self, state: EpidemicGraphState, policy_names: List[str], force: bool = False) -> EpidemicGraphState:
        metrics = state["metrics"]
        activated = list(state.get("activated_policies", []))
        logs = state.get("decision_log", [])

        for name in policy_names:
            policy = self._policy_from_name(name)
            if policy is None:
                continue
            if policy.policy_type in self.model.policy_manager.active_policies:
                continue

            score = 1.0 if force else self._score_policy(metrics, policy)
            if force or score >= self.decision_threshold:
                self.model.policy_manager.activate_policy(policy, state["step"])
                activated.append(name)
                logs.append(
                    {
                        "step": str(state["step"]),
                        "policy": name,
                        "confidence": "force" if force else f"{score:.2f}",
                    }
                )

        state["activated_policies"] = list(dict.fromkeys(activated))
        state["decision_log"] = logs
        return state

    def node_fetch_metrics(self, state: EpidemicGraphState) -> EpidemicGraphState:
        state["metrics"] = self._metrics()
        state["node_trace"] = state.get("node_trace", []) + ["fetch_metrics"]
        return state

    def node_predict(self, state: EpidemicGraphState) -> EpidemicGraphState:
        state["predictions"] = self._predictions()
        state["node_trace"] = state.get("node_trace", []) + ["predict"]
        return state

    def node_query_kb(self, state: EpidemicGraphState) -> EpidemicGraphState:
        metrics = state["metrics"]
        situation = f"Hospital occupancy {metrics['hospital_occupancy_rate']:.1%}. Untreated infected {metrics['infected_untreated']}."
        candidates = self.kb.extract_actionable_policies(situation, k=3)
        state["candidate_policies"] = [p.policy_type.value for p in candidates]
        state["node_trace"] = state.get("node_trace", []) + ["query_kb"]
        return state

    def node_decide(self, state: EpidemicGraphState) -> EpidemicGraphState:
        metrics = state["metrics"]
        candidates = list(dict.fromkeys(state.get("candidate_policies", [])))
        selected = self._heuristic_policy_selection(metrics, candidates)

        if metrics["hospital_occupancy_rate"] > 0.90 and len(self.model.policy_manager.active_policies) == 0:
            selected.extend(["mask_mandate", "social_distancing", "elective_surgery_ban"])

        selected = list(dict.fromkeys(selected))
        state["activated_policies"] = []
        state = self._activate_policy_set(state, selected, force=False)
        state["node_trace"] = state.get("node_trace", []) + ["decide"]
        return state

    def node_conflict_resolution(self, state: EpidemicGraphState) -> EpidemicGraphState:
        activated = state.get("activated_policies", [])
        if "school_closure" in activated and "resource_rationing" in activated:
            activated.remove("school_closure")
            state["decision_log"] = state.get("decision_log", []) + [
                {
                    "step": str(state["step"]),
                    "policy": "school_closure_removed",
                    "confidence": "rule",
                }
            ]
        state["activated_policies"] = activated
        state["node_trace"] = state.get("node_trace", []) + ["conflict_resolution"]
        return state

    def node_surge_policy(self, state: EpidemicGraphState) -> EpidemicGraphState:
        surge_order = [
            "crisis_standards",
            "resource_rationing",
            "elective_surgery_ban",
            "inter_hospital_transfer",
            "mask_mandate",
            "social_distancing",
            "school_closure",
        ]
        state = self._activate_policy_set(state, surge_order, force=True)
        state["interrupt_reason"] = "Crisis threshold reached: surge policy activation executed."
        state["node_trace"] = state.get("node_trace", []) + ["surge_policy"]
        return state

    def route_after_decision(self, state: EpidemicGraphState) -> str:
        if state["metrics"]["hospital_occupancy_rate"] > 0.95:
            return "surge"
        activated = state.get("activated_policies", [])
        if "school_closure" in activated and "resource_rationing" in activated:
            return "conflict"
        return "end"

    def _build_graph(self):
        builder = StateGraph(EpidemicGraphState)
        builder.add_node("fetch_metrics", self.node_fetch_metrics)
        builder.add_node("predict", self.node_predict)
        builder.add_node("query_kb", self.node_query_kb)
        builder.add_node("decide", self.node_decide)
        builder.add_node("conflict_resolution", self.node_conflict_resolution)
        builder.add_node("surge_policy", self.node_surge_policy)

        builder.add_edge(START, "fetch_metrics")
        builder.add_edge("fetch_metrics", "predict")
        builder.add_edge("predict", "query_kb")
        builder.add_edge("query_kb", "decide")

        builder.add_conditional_edges(
            "decide",
            self.route_after_decision,
            {
                "surge": "surge_policy",
                "conflict": "conflict_resolution",
                "end": END,
            },
        )
        builder.add_edge("conflict_resolution", END)
        builder.add_edge("surge_policy", END)

        return builder.compile()

    def run_graph_step(self, step: int) -> Dict[str, object]:
        initial: EpidemicGraphState = {"step": step, "decision_log": [], "node_trace": []}
        out = self.graph.invoke(initial)
        self.history.append(
            {
                "step": step,
                "trace": out.get("node_trace", []),
                "activated": out.get("activated_policies", []),
                "interrupt_reason": out.get("interrupt_reason", ""),
            }
        )
        return out
