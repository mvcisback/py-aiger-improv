from __future__ import annotations

import re
from functools import reduce
from typing import Callable, Sequence, Optional

import attr
import mdd
import numpy as np
import aiger_bdd
import aiger_bv as BV
import aiger_coins as C
import aiger_coins.infer
import aiger_discrete as D
import funcy as fn
from aiger_coins.pcirc import Distribution
from aiger_discrete import FiniteFunc
from aiger_discrete.mdd import to_mdd
from mdd.nx import to_nx


import aiger_improv


TIMED_NAME = re.compile(r"(.*)##time_(\d+)$")


@fn.curry
def causal_key(env_action, timed_name):
    name, time = TIMED_NAME.match(timed_name).groups()
    return int(time), (env_action == name)


@attr.s(frozen=True, auto_attribs=True)
class Model:
    mdd: mdd.DecisionDiagram
    order: Sequence[str]
    preimage: Callable[[BV.UnsignedBVExpr], BV.UnsignedBVExpr]
    coin_biases: Callable[[str], Optional[Sequence[float]]]
    steps: int
    dyn: Optional[C.PCirc] = None

    @property
    def true_sym(self):
        return self.mdd.interface.output.decode(True)

    @property
    def false_sym(self):
        return self.mdd.interface.output.decode(False)

    def override(self, expr, is_sat: bool=True):
        """Change satisfaction label of feature closure of expr."""
        preimg = self.preimage(expr)
        label = self.true_sym if is_sat else self.false_sym
        return attr.evolve(self, mdd=self.mdd.override(preimg, label))

    def is_random(self, name):
        return self.coin_biases(name) is not None

    def graph(self):
        return to_nx(self.mdd, reindex=False)

    def prob(self, guard, *, log=False):
        assert all(self.is_random(i) for i in guard.inputs)

        order = list(guard.inputs)
        guard = guard.bundle_inputs('##coins', order=order)
        coin_biases = fn.lmapcat(self.coin_biases, order)

        pcirc = C.PCirc(
            circ=guard.aigbv, 
            coin_biases=len(order) * coin_biases,
            coins_id='##coins',
        )
        return C.infer.prob(pcirc, log=log)

    def size(self, guard):
        return aiger_bdd.count(guard)

    def time_step(self, name):
        if isinstance(name, bool):
            return self.steps

        _, time = TIMED_NAME.match(name).groups()
        return int(time)

    @property
    def actions(self):
        return tuple(name for name in self.order if not self.is_random(name))

    def improviser(self, *, rationality=None, psat=None):
        if (rationality is not None) and (psat is not None):
            raise ValueError("rationality and psat are mutually exclusive.")
        if rationality is not None:
            return aiger_improv.improviser(self, rationality)
        return aiger_improv.fit(self, psat)

    def find_coins(self, state, action, observation):
        features, state2 = observation
        action = {fn.first(self.dyn.inputs): action}
        return C.infer.find_coins(
            self.dyn,
            inputs=action, outputs=features,
            latchins=state, latchouts=state2,
        )


def onehot_gadget(output: str):
    sat = BV.uatom(1, output)
    false, true = BV.uatom(2, 0b01), BV.uatom(2, 0b10)
    expr = BV.ite(sat, true, false) \
             .with_output('sat')
    
    encoder = D.Encoding(
        encode=lambda x: 1 << int(x),
        decode=lambda x: bool((x >> 1) & 1),
    )

    return D.from_aigbv(
        expr.aigbv,
        output_encodings={'sat': encoder},
    )


def from_pcirc(dyn: C.PCirc, monitor, steps: int):
    if dyn.outputs != monitor.inputs:
        raise ValueError("Monitor needs to match dynamics interface!")

    if len(dyn.inputs) != 1:
        raise NotImplementedError("Currently single input dynamics required.")

    monitor = dyn >> monitor
    if len(monitor.outputs) != 1:
        raise ValueError("Only support single output monitors at the moment.")
    monitor >>= onehot_gadget(fn.first(monitor.outputs))

    unrolled_d: FiniteFunc = dyn.circ.unroll(steps)
    unrolled_m: FiniteFunc = monitor.circ.unroll(steps, only_last_outputs=True)

    key = causal_key(monitor.coins_id)
    causal_order = tuple(sorted(unrolled_m.inputs, key=key))

    def coin_biases(name):
        _, is_env = key(name)
        biases = monitor.coin_biases if is_env else ()
        return None if len(biases) == 0 else biases

    omap = unrolled_d.omap

    def preimage(expr):
        assert expr.inputs <= unrolled_d.outputs

        circ = expr.aigbv
        for name in unrolled_d.outputs - expr.inputs:
            circ >>= BV.sink(omap[name].size, [name])

        valid = BV.uatom(1, unrolled_d.valid_id)
        sat = BV.uatom(1, expr.output)

        preimg = (unrolled_d >> circ).aigbv >> (sat & valid).aigbv
        assert preimg.inputs == unrolled_d.inputs

        return BV.UnsignedBVExpr(preimg)

    mdd = to_mdd(unrolled_m, order=causal_order)
    mdd.bdd.bdd.configure(reordering=False)  #  TODO: to_mdd should do this.

    return Model(
        preimage=preimage,
        mdd=to_mdd(unrolled_m, order=causal_order),
        coin_biases=coin_biases,
        order=causal_order,
        steps=steps,
        dyn=dyn,
    )


__all__ = ['Model', 'from_pcirc']
