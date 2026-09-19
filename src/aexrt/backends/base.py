from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from ..graph import Graph
from ..execution import BackendCapabilities, ExecutionPlan, compile_execution_plan


@dataclass(frozen=True)
class BackendInfo:
    name: str
    device: str
    capabilities: Dict[str, Any]

    def structured_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.from_dict(self.capabilities)


class Backend(ABC):
    name = "backend"
    supported_ops = frozenset[str]()
    supported_dtypes = frozenset[str]()
    supported_features = frozenset[str]()

    @abstractmethod
    def info(self) -> BackendInfo: ...

    @abstractmethod
    def prepare(self, graph: Graph) -> None: ...

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

    def compile_plan(self, graph: Graph) -> ExecutionPlan:
        return compile_execution_plan(graph, self.info())
