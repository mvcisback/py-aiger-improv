from __future__ import annotations

import random
from functools import reduce
from typing import Mapping, Sequence

import aiger_bdd as B
import aiger
import aiger_bv as BV
import attr
import funcy as fn
import networkx as nx
import numpy as np
import scipy as sp
from bdd2dfa.b2d import BNode
from scipy.optimize import root_scalar

from aiger_improv.model import Model, TIMED_NAME
from aiger_improv.dist import Distribution


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


def improviser(model, rationality):
    graph = model.graph()
    entropy_bump = entropy_bumper(model=model, graph=graph)

    node2val = {}
    for node in nx.topological_sort(graph.reverse()):
        data = graph.nodes[node]
        name = data['label']

        if isinstance(name, bool):                        # Leaf Case
            data['val'] = float(name) * rationality
            data['lsat'] = 0 if name else -float('inf')
            continue

        kids = list(graph.neighbors(node))
        values = amap(lambda k: graph.nodes[k]['val'], kids)
        values += entropy_bump(node)   # TODO: arguably a model bug.

        if model.is_random(name):                    # Chance Case.
            probs = kid_probs(model, graph, node)
            data['val'] = (probs * values).sum()
        else:                                        # Decision Case.
            probs = sp.special.softmax(values)
            data['val'] = sp.special.logsumexp(values)

        for prob, kid in zip(probs, kids):           # Record probability.
            graph.edges[node, kid]['prob'] = prob

        # Update log satisfaction probability.
        lsats = np.array([graph.nodes[k]['lsat'] for k in kids])
        data['lsat'] = sp.special.logsumexp(lsats, b=probs)
            
    assert graph.in_degree(node) == 0, "Should finish at root."
    return Improviser(root=node, graph=graph, model=model)


def fit(model: Model, psat: float, top: float = 100) -> Improviser:
    """Fit a max causal ent policy with satisfaction probability psat."""
    assert 0 <= psat <= 1

    actor = None

    def f(coeff):
        nonlocal actor
        actor = improviser(model, coeff)
        return actor.sat_prob() - psat

    if f(-top) > 0:
        coeff = 0
    elif f(top) < 0:
        coeff = top
    else:
        res = root_scalar(f, bracket=(-top, top), method="brentq")
        coeff = res.root

    if coeff < 0: # More likely the negated spec than this one.
        coeff = 0

    return actor


@attr.s(frozen=True, auto_attribs=True)
class Improviser:
    root: BNode
    graph: nx.DiGraph
    model: Model

    def sat_prob(self, log=False):
        lsat = self.graph.nodes[self.root]['lsat']
        return lsat if log else np.exp(lsat)

    @property
    def node2val(self) -> Mapping[BNode, float]:
        return {n: d['val'] for n, d in self.graph.nodes(data=True)}

    def transition(self, state, action):
        curr_name = self.graph.nodes[state]['label']
        action = self.model.mdd.io.var(curr_name).encode(action)

        if isinstance(curr_name, bool):
            return state  # Self loop on leaf.

        assert state.node.var.startswith(curr_name)
        for kid in self.graph.neighbors(state):  # Find the transition.
            guard = self.graph.edges[state, kid]['label']
            assert guard.inputs == {curr_name}            

            if guard({curr_name: action})[0]:
                return kid
        raise ValueError("Invalid transition.")

    def run(self):
        state = self.root
        measure = self.model.size
        for curr_name in self.model.order:
            state_name = self.graph.nodes[state]['label']
            # ----- Provide action distribution -----
            if curr_name != state_name:  # Uniformly select from valid.
                dist = [(self.model.mdd.io.var(curr_name).valid, 1)]
            else:
                kids = self.graph.neighbors(state)
                edges = (self.graph.edges[state, k] for k in kids)
                dist = [(edge['label'], edge['prob']) for edge in edges]

            action = yield Distribution(dist, model=self.model, name=curr_name)
            state = self.transition(state, action)
        raise ValueError("Action sent after episode ended.")
  
    def prob(self, trc, log: bool = False) -> float:
        runner, lprob, obs = self.run(), 0, None 
        for sym in trc:
            action_dist, obs = runner.send(obs), sym
            lprob += action_dist.logprob(sym)

        return lprob if log else np.exp(lprob)

    def sample(self, seed=None):
        random.seed(seed)
        runner, action = self.run(), None
        for _ in self.model.order:
            action_dist = runner.send(action)
            action = action_dist.sample()
            yield action

    def policy(self, seed=None):
        actions = self.model.actions
        if actions == self.model.order:  # Deterministic Case.
            yield from self.sample(seed)
            return

        random.seed(seed)
        runner, env_action = self.run(), None
        for _ in actions:
            action_dist = runner.send(env_action)
            action = action_dist.sample()
            env_action = yield action


__all__ = ['Improviser', 'improviser']
