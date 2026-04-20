from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class State:
    SUSCEPTIBLE = 0
    INFECTED_UNTREATED = 1
    HOSPITALIZED = 2
    RECOVERED = 3


class PolicyType(Enum):
    SOCIAL_DISTANCING = "social_distancing"
    ELECTIVE_SURGERY_BAN = "elective_surgery_ban"
    MASK_MANDATE = "mask_mandate"
    SCHOOL_CLOSURE = "school_closure"
    INTER_HOSPITAL_TRANSFER = "inter_hospital_transfer"
    CRISIS_STANDARDS = "crisis_standards"
    RESOURCE_RATIONING = "resource_rationing"


@dataclass
class Policy:
    policy_type: PolicyType
    active: bool
    effectiveness: float
    implementation_cost: int
    duration_steps: int
    description: str


@dataclass
class Prediction:
    target_metric: str
    current_value: float
    predicted_value: float
    prediction_horizon: int
    confidence: float


class PolicyManager:
    def __init__(self):
        self.active_policies: Dict[PolicyType, Policy] = {}
        self.policy_history: List[Tuple[int, PolicyType, bool]] = []

    def activate_policy(self, policy: Policy, step: int) -> None:
        self.active_policies[policy.policy_type] = policy
        self.policy_history.append((step, policy.policy_type, True))

    def get_transmission_modifier(self) -> float:
        modifier = 1.0
        if PolicyType.SOCIAL_DISTANCING in self.active_policies:
            modifier *= 1 - self.active_policies[PolicyType.SOCIAL_DISTANCING].effectiveness
        if PolicyType.MASK_MANDATE in self.active_policies:
            modifier *= 1 - self.active_policies[PolicyType.MASK_MANDATE].effectiveness
        return modifier

    def get_hospital_capacity_modifier(self) -> float:
        modifier = 1.0
        if PolicyType.ELECTIVE_SURGERY_BAN in self.active_policies:
            modifier += self.active_policies[PolicyType.ELECTIVE_SURGERY_BAN].effectiveness
        return modifier
