from app.agents.supervisor import SupervisorAgent


def test_route_request() -> None:
    supervisor = SupervisorAgent()

    assert supervisor.route_request("lead") == "lead_agent"
    assert supervisor.route_request("quote") == "insurance_agent"
