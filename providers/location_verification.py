from enum import Enum
from typing import Protocol

from execution import RunBudget


class CountryMatch(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class CityCountryVerifier(Protocol):
    def verify(
        self,
        city: dict,
        expected_country: str,
        *,
        budget: RunBudget | None = None,
    ) -> CountryMatch:
        """Match the same city entity, not merely its name, to the country.

        Return UNKNOWN when identity or country evidence is insufficient.
        Network implementations must count attempts against the shared budget.
        """
        ...
