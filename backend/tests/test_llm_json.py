from types import SimpleNamespace

from app.llm_json import invoke_json_model
from app.models import AgentPlan


class FakeJsonLlm:
    def __init__(self, content: str) -> None:
        self.content = content
        self.response_format = None

    def bind(self, **kwargs):
        self.response_format = kwargs.get("response_format")
        return self

    def invoke(self, _messages):
        return SimpleNamespace(content=self.content)


def test_json_mode_returns_validated_plan() -> None:
    llm = FakeJsonLlm(
        '一些无关前缀 {"intent":"trip_query","objective":"解释行程",'
        '"tools":["trip"],"planned_steps":["trip"]} 结尾'
    )

    result = invoke_json_model(llm, AgentPlan, [])

    assert result is not None
    assert result.intent == "trip_query"
    assert llm.response_format == {"type": "json_object"}


def test_json_mode_retries_invalid_response() -> None:
    result = invoke_json_model(FakeJsonLlm("not json"), AgentPlan, [], attempts=1)

    assert result is None
