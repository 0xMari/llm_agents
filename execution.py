from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchLimits(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    max_destinations: int = Field(default=1, ge=1, le=3)
    max_periods_per_destination: int = Field(default=1, ge=1)
    max_outbound: int = Field(default=1, ge=1)
    max_serpapi_calls: int = Field(default=5, ge=0)
    max_llm_calls: int = Field(default=2, ge=0)


class SearchLimitExceeded(RuntimeError):
    def __init__(self, service: str, used: int, limit: int):
        self.service = service
        self.used = used
        self.limit = limit
        super().__init__(f"{service} call limit reached: {used}/{limit}")


@dataclass
class RunBudget:
    limits: SearchLimits
    used: dict[str, int] = field(
        default_factory=lambda: {"serpapi": 0, "llm": 0}, init=False
    )

    def claim(self, service: Literal["serpapi", "llm"]) -> None:
        limit = {
            "serpapi": self.limits.max_serpapi_calls,
            "llm": self.limits.max_llm_calls,
        }[service]
        if self.used[service] >= limit:
            raise SearchLimitExceeded(service, self.used[service], limit)
        # Count attempts, including timeouts; a failed call is not refunded.
        self.used[service] += 1
