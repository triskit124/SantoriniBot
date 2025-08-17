from typing import Literal

from .Player import Agent


AGENT_CHOICES: dict[str, type[Agent]] = {}

from .HumanAgent import HumanAgent
AGENT_CHOICES["human"] = HumanAgent

from .RandomAgent import RandomAgent
AGENT_CHOICES["random"] = RandomAgent

from .ForwardSearchAgent import ForwardSearchAgent
AGENT_CHOICES["forward_search"] = ForwardSearchAgent

from .MiniMaxAgent import MiniMaxAgent
AGENT_CHOICES["mini_max"] = MiniMaxAgent

try:
    from .NNAgent import NNAgent
    AGENT_CHOICES["neural_net"] = NNAgent
except(ImportError):
    pass

AgentType = Literal[AGENT_CHOICES.keys()]

TYPE_CHECKING = False


