from __future__ import annotations

from functools import reduce

import aiger_bdd as B
import aiger
import aiger_bv as BV
import attr
import funcy as fn
import networkx as nx
import numpy as np

from aiger_improv.model import Model


def amap(func, collection) -> np.array:
    return np.array(fn.lmap(func, collection))


TRUE = BV.uatom(1, 1)


def entropy_bumper(model, graph):
    """
    Account for # decisions on edge by expanding guards.
    Note: This is arguably a bug in the model.
    """
    actions = model.actions

    def func(node):
        # TODO: if is chance, shouldn't include initial guard!
        name = graph.nodes[node]['label']
        curr_time = model.time_step(name)

        decisions = []
        for kid in graph.neighbors(node):
            guard = graph.edges[node, kid]['label']
            kid_name = graph.nodes[kid]['label']
            kid_time = model.time_step(kid_name)

            skipped_actions = actions[curr_time + 1:kid_time]
            
            skipped = (model.mdd.io.var(name).valid for name in skipped_actions)
            guard = reduce(lambda x, y: x & y, skipped, TRUE)

            decisions.append(guard)

        sizes = amap(model.size, decisions)
        return np.log(sizes)
    return func


def kid_probs(model, graph, node):
    kids = graph.neighbors(node)
    guards = (graph.edges[node, k]['label'] for k in kids)
    return amap(model.prob, guards)


@attr.s(frozen=True, auto_attribs=True)
class Improviser:
    model: Model
    rationality: float

    def values(self):
        graph = self.model.graph()
        entropy_bump = entropy_bumper(model=self.model, graph=graph)

        value = {}
        for node in nx.topological_sort(graph.reverse()):
            name = graph.nodes[node]['label']

            if isinstance(name, bool):                        # Leaf Case
                value[node] = float(name) * self.rationality
                continue

            kids = graph.neighbors(node)
            values = amap(value.get, kids)
            values += entropy_bump(node)   # TODO: arguably a model bug.

            if self.model.is_random(name):                    # Chance Case.
                probs = kid_probs(self.model, graph, node)
                value[node] = (probs * values).sum()
            else:                                             # Decision Case.
                value[node] = np.logaddexp(*values)

        return value


__all__ = ['Improviser']
