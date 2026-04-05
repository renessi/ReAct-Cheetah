from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError