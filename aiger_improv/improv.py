from __future__ import annotations

import aiger_bdd as B
import attr
import networkx as nx
import numpy as np

from aiger_improv.model import Model


def amap(func, collection) -> np.array:
    return np.array(fn.lmap(func, collection))


@attr
class Improviser:
    model: Model
    rationality: float

    def values(self):
        value = {}
        graph = self.model.graph()
        for node in nx.topological_sort(graph.reverse()):
            name = graph.nodes[node]['label']

            if isinstance(name, bool):                        # Leaf
                value[node] = float(name) * self.rationality
                continue

            kids = graph.neighbors(node)
            values = amap(value.get, kids)
            guards = [graph.edges[node, k]['label'] for k in kids]

            if self.model.is_random(name):                     # Chance
                probs = amap(self.model.prob, guards)
            else:                                              # Decision
                # Account for # decisions on edge by expanding guards.
                # Note: This is arguably a bug in the model.
                decisions = []
                # TODO: optimize to not depend on length of horizon.
                for kid, guard in zip(kids, guards):
                    # TODO guard |= 
                    decisions.append(guard)

                curr_time = self.model.time_step(name)
                skipped = amap(self.model.time_step, kid_names) - curr_time - 1
                sizes = amap(model.size, guards)

                pass
            

