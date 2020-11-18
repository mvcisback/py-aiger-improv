import random
from typing import Callable, Sequence, Tuple

import aiger_bv as BV
import aiger_bdd
import attr
import funcy as fn
import mdd
import numpy as np

from aiger_improv.model import Model


@attr.s(auto_attribs=True, frozen=True)
class Distribution:
    data: Sequence[Tuple[BV.UnsignedBVExpr, float]]
    name: str
    model: Model

    @property
    def is_decision_variable(self):
        return not self.model.is_random(self.name)

    @property
    def variable(self):
        return self.model.mdd.io.var(self.name)

    @property
    def encode(self):
        return self.variable.encode

    @property
    def decode(self):
        return self.variable.encode

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
        return self.decode(bits)

    def logprob(self, action):
        action = self.encode(action)
        lprob = -float('inf')
        for meta_action, prob in self.data:  # Linear scan for meta-action.
            if not meta_action({self.name: action})[0]:
                continue

            lprob = np.log(prob)  # Meta-action prob.
            if self.is_decision_variable:    # Uniform decision meta-action.
                lprob -= np.log(self.model.size(meta_action))
            else:                            # Flip bias coins for env action.
                query = self.variable.expr() == action
                lprob += np.log(self.model.prob(query))
        return lprob
