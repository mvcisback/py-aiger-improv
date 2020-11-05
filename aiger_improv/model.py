from __future__ import annotations

import re
from typing import Callable, Sequence


import attr
import mdd
import numpy as np
import aiger_bv as BV
import aiger_coins as C
import aiger_discrete as D
import funcy as fn
from aiger_coins.pcirc import Distribution
from aiger_discrete import FiniteFunc
from aiger_discrete.mdd import to_mdd
from pyrsistent import pmap
from pyrsistent.typing import PMap


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
    coin_biases: Callable[[str], Sequence[float]]  # Biases of a given input.

    @property
    def true_sym(self):
        return self.mdd.interface.output.decode(1)

    @property
    def false_sym(self):
        return self.mdd.interface.output.decode(0)

    def override(self, expr, is_sat: bool=True):
        """Change satisfaction label of feature closure of expr."""
        expr = self.preimage(expr)
        label = self.true_sym if is_sat else self.false_sym
        model = self.model.override(expr, label)
        return attr.evolve(self, model=model)

    def is_coin(self, name):
        return self.coin_biases(name) != ()


def from_pcirc(dyn: C.PCirc, monitor, steps: int):
    if dyn.outputs != monitor.inputs:
        raise ValueError("Monitor needs to match dynamics interface!")

    monitor = dyn >> monitor
    if len(monitor.outputs) != 1:
        raise ValueError("Only support single output monitors at the moment.")

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
    )
