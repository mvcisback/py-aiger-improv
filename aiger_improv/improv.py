from __future__ import annotations

import aiger_bdd as B
import attr
import networkx as nx
import numpy as np
from scipy.special import logsumexp

from aiger_improv.model import Model


def amap(func, collection) -> np.array:
    return np.array(fn.lmap(func, collection))


@attr
class Improviser:
    model: Model
    rationality: float

    def values(self):
        actions = self.model.actions
        value = {}
        graph = self.model.graph()
        for node in nx.topological_sort(graph.reverse()):
            name = graph.nodes[node]['label']

            if isinstance(name, bool):                        # Leaf Case
                value[node] = float(name) * self.rationality
                continue

            kids = graph.neighbors(node)
            values = amap(value.get, kids)
            guards = [graph.edges[node, k]['label'] for k in kids]

            if self.model.is_random(name):                    # Chance Case
                probs = amap(self.model.prob, guards)
                value[node] = (probs * values).sum()
                continue

            # ------------  Decision Case -------------------

            # Account for # decisions on edge by expanding guards.
            # Note: This is arguably a bug in the model.
            curr_time = self.model.time_step(name)

            decisions = []
            for kid, guard in zip(kids, guards):
                kid_time = self.model.time_step(name)
                skipped = actions[curr_time+1:kid_time]
                for name in skipped:  # Expand each guard.
                    guard |= self.model.mdd.io.var(name).valid
                decisions.append(guard)

            # Apply expanded guard entropy bump.
            values += np.log(amap(model.size, guards))
            value[node] = logsumexp(values)

        return value
