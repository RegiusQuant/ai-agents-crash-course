import logging
from collections import deque
from typing import List, Optional, Any

# Module‐level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Crew:
    """
    Manages a collection of agents, their dependencies, and provides
    methods to sort, visualize, and execute them in dependency order.
    """

    _active: Optional["Crew"] = None

    def __init__(self):
        self._agents: List[Any] = []

    def __enter__(self) -> "Crew":
        Crew._active = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Crew._active = None

    @classmethod
    def register(cls, agent: Any) -> None:
        """
        Register an agent with the currently active crew context.
        """
        if cls._active is None:
            logger.warning(
                "No active Crew context to register agent %r",
                getattr(agent, "name", agent),
            )
            return
        cls._active.add(agent)

    def add(self, agent: Any) -> None:
        """
        Add an agent to this crew.
        """
        if not hasattr(agent, "name") or not hasattr(agent, "run"):
            raise TypeError(
                "Agent must have 'name' attribute and 'run()' method")
        self._agents.append(agent)
        logger.debug("Agent %s added to crew", agent.name)

    def topological_sort(self) -> List[Any]:
        """
        Return agents in dependency order, raising on cycles.
        """
        # build in-degree map
        in_degree = {a: len(getattr(a, "dependencies", []))
                     for a in self._agents}
        queue = deque(a for a, deg in in_degree.items() if deg == 0)
        sorted_list: List[Any] = []

        while queue:
            node = queue.popleft()
            sorted_list.append(node)
            for dep in getattr(node, "dependents", []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(sorted_list) != len(self._agents):
            logger.error(
                "Cycle detected among agents: %s",
                [getattr(a, "name", str(a)) for a in self._agents],
            )
            raise ValueError("Circular dependency detected")
        return sorted_list

    def run_all(self) -> None:
        """
        Execute each agent in dependency order, logging their outputs.
        """
        try:
            for agent in self.topological_sort():
                print(f"\n{'-'*50}\nRunning agent {agent.name}\n{'-'*50}\n")
                result = agent.run()
                print(
                    f"\n{'-'*50}\nResult from {agent.name}: {result}\n{'-'*50}\n")
        except Exception as e:
            logger.exception("Error during crew execution: %s", e)
            raise
