import random
from typing import Callable, Sequence, Tuple

import aiger_bv as BV
import aiger_bdd
import attr
import funcy as fn
import mdd
import numpy as np


@attr.s(auto_attribs=True, frozen=True)
class Distribution:
    data: Sequence[Tuple[BV.UnsignedBVExpr, float]]
    measure: Callable[[BV.UnsignedBVExpr], int]
    variable: mdd.Variable

    def sample(self):
        guard, *_ = random.choices(*zip(*self.data), k=1)
        bexpr, manager, *_ = aiger_bdd.to_bdd(guard)
        if bexpr == manager.false:
            raise ValueError('None zero probability assigned to empty set.')

        # Uniformly sample bits and use BDD to error-correct to a model.
        nbits = len(manager.vars)
        bits: int = random.getrandbits(nbits)  # Packed random bits.
        for i in range(nbits):
            var = bexpr.bdd.var_at_level(i)
            decision = bool((bits >> i) & 1)   # Look up ith random bit.
            bexpr2 = bexpr.let(**{var: decision})

            if bexpr2 == manager.false:
                bexpr2 = bexpr.let(**{var: not decision})
                bits ^= 1 << i  # Error correction.
            bexpr = bexpr2
        return self.variable.decode(bits)

    def logprob(self, sym):
        sym = self.variable.encode(sym)
        for guard, prob in self.data:  # Linear scan for action.
            if guard({self.variable.name: sym})[0]:
                #   ln[P(meta action) * P(action | meta action)]
                return np.log(prob) - np.log(self.measure(guard))
        return -float('inf')
