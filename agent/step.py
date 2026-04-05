from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None
