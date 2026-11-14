from __future__ import annotations


class SupervisorAgent:
    def route_request(self, intent: str) -> str:
        intents = {
            "lead": "lead_agent",
            "quote": "insurance_agent",
            "policy": "insurance_agent",
        }
        return intents.get(intent.lower(), "lead_agent")
