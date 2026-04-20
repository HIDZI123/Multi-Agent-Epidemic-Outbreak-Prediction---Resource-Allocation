from typing import List

from langchain_core.documents import Document

from core.policy import Policy, PolicyType


KNOWLEDGE_DOCS = {
    "actionable_hospital_protocols.txt": """
TRIGGER: Hospital occupancy > 80%
ACTIONS:
- ACTIVATE elective_surgery_ban (effectiveness: 0.25, cost: 100)
- INITIATE inter_hospital_transfer (effectiveness: 0.15, cost: 50)

TRIGGER: Hospital occupancy > 90%
ACTIONS:
- ACTIVATE crisis_standards (effectiveness: 0.20, cost: 200)
- IMPLEMENT resource_rationing (effectiveness: 0.10, cost: 150)
""",
    "community_intervention_protocols.txt": """
TRIGGER: Untreated infected > 100 OR transmission rate > 0.4
ACTIONS:
- IMPLEMENT social_distancing (effectiveness: 0.35, cost: 300)
- ACTIVATE mask_mandate (effectiveness: 0.20, cost: 100)
""",
    "resource_optimization_protocols.txt": """
TRIGGER: Multiple hospitals > 85% capacity
ACTIONS:
- COORDINATE inter_hospital_transfer (effectiveness: 0.25, cost: 100)

TRIGGER: Predicted ventilator shortage
ACTIONS:
- IMPLEMENT resource_rationing (effectiveness: 0.20, cost: 200)
""",
}


class EnhancedKnowledgeBase:
    def __init__(self):
        self.docs = [Document(page_content=txt, metadata={"source": name}) for name, txt in KNOWLEDGE_DOCS.items()]

    def search(self, query: str, k: int = 3) -> List[Document]:
        # Deterministic keyword matching keeps the GUI lightweight and reproducible.
        query_lower = query.lower()
        scored = []
        for doc in self.docs:
            score = 0
            for token in ["occupancy", "infected", "transmission", "hospital", "shortage"]:
                if token in query_lower and token in doc.page_content.lower():
                    score += 1
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]

    def extract_actionable_policies(self, situation: str, k: int = 3) -> List[Policy]:
        mapping = {
            "elective_surgery_ban": (PolicyType.ELECTIVE_SURGERY_BAN, 0.25, 100),
            "social_distancing": (PolicyType.SOCIAL_DISTANCING, 0.35, 300),
            "mask_mandate": (PolicyType.MASK_MANDATE, 0.20, 100),
            "school_closure": (PolicyType.SCHOOL_CLOSURE, 0.20, 350),
            "crisis_standards": (PolicyType.CRISIS_STANDARDS, 0.20, 200),
            "resource_rationing": (PolicyType.RESOURCE_RATIONING, 0.10, 150),
            "inter_hospital_transfer": (PolicyType.INTER_HOSPITAL_TRANSFER, 0.15, 80),
        }

        found: List[Policy] = []
        for result in self.search(situation, k=k):
            content = result.page_content.lower()
            for key, values in mapping.items():
                if key in content:
                    ptype, eff, cost = values
                    found.append(Policy(ptype, False, eff, cost, 20, f"KB match: {key}"))

        uniq = {}
        for policy in found:
            uniq[policy.policy_type] = policy
        return list(uniq.values())
