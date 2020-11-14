import pytest

import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as LTL
import funcy as fn
import numpy as np

from aiger_improv.model import from_pcirc
from aiger_improv.improv import Improviser


def test_never_false1():
    spec = LTL.atom('x').historically()
    monitor = BV.aig2aigbv(spec.aig)
    dyn = C.pcirc(BV.identity_gate(1, 'x'))
    
    horizon = 4
    model = from_pcirc(dyn, monitor, steps=horizon)

    coeff = np.log(2)
    actor = Improviser(model, coeff)
    
    expected = [
        0,
        coeff,
        np.log(1 + 2),
        np.log(3 + 2),
        np.log(7 + 2),
        np.log(15 + 2),
    ]
    # LSE visits powers of 2 for the special coeff.
    expected = fn.lmap(pytest.approx, expected)

    vals = sorted(list(actor.values().values()))
    assert all(x == y for x, y in zip(expected, vals))


def test_never_false_redemption():
    spec = LTL.atom('x').historically()
    monitor = BV.aig2aigbv(spec.aig)

    # Environment can save you.
    x, y = BV.uatom(1, 'x'), BV.uatom(1, 'y')
    xy = (x | y).with_output('x')  # env y can override x.

    dyn = C.pcirc(xy.aigbv) \
           .randomize({'y': {0: 0.75, 1: 0.25}})
    
    horizon = 3
    model = from_pcirc(dyn, monitor, steps=horizon)

    coeff = np.log(2)  # Special coeff to make LSE visit powers of 2.
    actor = Improviser(model, coeff)

    
    v8 = coeff
    v7 = 0
    v6 = coeff / 4
    v5 = np.logaddexp(v6, coeff)
    v4 = (np.log(8) + v5) / 4
    v3 = np.logaddexp(v4, v5)
    v2 = (3*np.log(4) + v3) / 4
    v1 = np.logaddexp(v2, v3)

    expected = sorted([v8, v7, v6, v5, v4, v3, v2, v1])
    expected = fn.lmap(pytest.approx, expected)

    vals = sorted(list(actor.values().values()))

    for x, y in zip(vals, expected):
        assert x == y

