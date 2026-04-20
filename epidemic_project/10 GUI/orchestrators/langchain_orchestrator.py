import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.analytics import PredictiveAnalytics
from core.knowledge import EnhancedKnowledgeBase
from core.policy import Policy, PolicyType
from core.simulation import EpidemicModel

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


@dataclass
class Decision:
    step: int
    agent_name: str
    action: str
    confidence: float
    reasoning: str

try:
    llm_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    llm_api_key = os.environ.get("GROQ_API_KEY", "your-groq-api-key")

class LangChainOrchestrator:
    def __init__(
        self,
        model: EpidemicModel,
        kb: EnhancedKnowledgeBase,
        analytics: PredictiveAnalytics,
        llm_model_id: str = "llama-3.3-70b-versatile",
        llm_api_key: str = llm_api_key,
        llm_base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.2,
    ):
        self.model = model
        self.kb = kb
        self.analytics = analytics
        self.decision_log: List[Decision] = []
        self.has_llm = False
        self.policy_chain = None

        if llm_api_key and ChatOpenAI is not None:
            try:
                llm = ChatOpenAI(
                    model=llm_model_id,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                    temperature=temperature,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "Return strict JSON list of policy names from: social_distancing, mask_mandate, elective_surgery_ban, crisis_standards, inter_hospital_transfer, resource_rationing, school_closure.",
                        ),
                        (
                            "human",
                            "Metrics: {metrics}. Predictions: {predictions}. KB candidates: {kb_actions}. JSON only.",
                        ),
                    ]
                )
                self.policy_chain = prompt | llm | StrOutputParser()
                self.has_llm = True
            except Exception:
                self.has_llm = False

    def _predictions_snapshot(self) -> Dict[str, Dict[str, float]]:
        if len(self.model.datacollector.model_vars) < 15:
            return {}
        df = self.model.datacollector.get_model_vars_dataframe()
        hosp = self.analytics.predict_hospital_demand(df)
        peak = self.analytics.predict_infection_peak(df)
        return {"hospital_demand": hosp.__dict__, "infection_peak": peak.__dict__}

    def _fallback_policy_selection(self, metrics: Dict[str, float], kb_policies: List[Policy]) -> List[str]:
        selected = []
        if metrics["hospital_occupancy_rate"] > 0.55:
            selected.extend(["elective_surgery_ban", "inter_hospital_transfer"])
        if metrics["infected_untreated"] > 35 or metrics["effective_transmission_rate"] > 0.20:
            selected.extend(["social_distancing", "mask_mandate"])
        selected.extend([p.policy_type.value for p in kb_policies[:2]])
        return list(dict.fromkeys(selected))

    def _parse_policy_output(self, text: str) -> List[str]:
        try:
            payload = json.loads(text)
            values = payload.get("policies", [])
            return [v for v in values if isinstance(v, str)]
        except Exception:
            return []

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
        policy_type, eff, cost = mapping[name]
        return Policy(policy_type, False, eff, cost, 20, f"Activated by LangChain flow: {name}")

    def evaluate_and_act(self, step: int, decision_threshold: float = 0.35) -> Dict[str, object]:
        metrics = self.model.metrics_snapshot()
        predictions = self._predictions_snapshot()
        situation = f"Hospital occupancy {metrics['hospital_occupancy_rate']:.1%}. Untreated infected {metrics['infected_untreated']}."
        kb_policies = self.kb.extract_actionable_policies(situation, k=3)

        if self.has_llm and self.policy_chain is not None:
            raw_policy = self.policy_chain.invoke(
                {
                    "metrics": json.dumps(metrics),
                    "predictions": json.dumps(predictions),
                    "kb_actions": [p.description for p in kb_policies],
                }
            )
            policy_names = self._parse_policy_output(raw_policy)
        else:
            policy_names = self._fallback_policy_selection(metrics, kb_policies)

        activated = []
        for name in policy_names:
            candidate = self._policy_from_name(name)
            if candidate is None:
                continue
            if candidate.policy_type in self.model.policy_manager.active_policies:
                continue

            urgency = 0.0
            urgency += 0.5 if metrics["hospital_occupancy_rate"] > 0.85 else 0.0
            urgency += 0.3 if metrics["infected_untreated"] > 80 else 0.0
            urgency += 0.2 if metrics["effective_transmission_rate"] > 0.3 else 0.0
            score = min(1.0, urgency * 0.5 + candidate.effectiveness * 0.3 + (1 / (1 + candidate.implementation_cost / 100)) * 0.2)

            if score >= decision_threshold:
                self.model.policy_manager.activate_policy(candidate, step)
                activated.append(candidate.policy_type.value)
                reason = f"occupancy={metrics['hospital_occupancy_rate']:.2f}, infected={metrics['infected_untreated']}"
                self.decision_log.append(Decision(step, "PolicyStrategist", f"ACTIVATE_{candidate.policy_type.value}", score, reason))

        return {
            "step": step,
            "metrics": metrics,
            "predictions": predictions,
            "activated": activated,
            "llm_enabled": self.has_llm,
        }
