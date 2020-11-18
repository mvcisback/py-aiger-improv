import pytest

import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as LTL
import funcy as fn
import numpy as np

from aiger_improv.model import from_pcirc
from aiger_improv.improv import improviser, fit


def test_never_false1():
    spec = LTL.atom('x').historically()
    monitor = BV.aig2aigbv(spec.aig)
    dyn = C.pcirc(BV.identity_gate(1, 'x'))
    
    horizon = 4
    model = from_pcirc(dyn, monitor, steps=horizon)

    coeff = np.log(2)
    actor = improviser(model, coeff)
    
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

    vals = sorted(list(actor.node2val.values()))
    assert all(x == y for x, y in zip(expected, vals))

    expected = sorted([
        np.log(9) - np.log(17),
        np.log(8) - np.log(17),
    ])
    expected = fn.lmap(pytest.approx, expected)

    def lprob(elems):
        return actor.prob(elems, log=True)

    assert lprob([]) == 0
    assert lprob([1]) == pytest.approx(np.log(9) - np.log(17))
    assert lprob([1, 1]) == pytest.approx(np.log(5) - np.log(17))
    assert lprob([1, 1, 1]) == pytest.approx(np.log(3) - np.log(17))
    assert lprob([1, 1, 1, 1]) == pytest.approx(coeff - np.log(17))

    # ----------- Fail on first state ---------------------
    base = np.log(8) - np.log(17)
    assert lprob([0]) == pytest.approx(base)
    # Uniform after failing.
    assert lprob([0, 1]) == pytest.approx(base - np.log(2))
    assert lprob([0, 0]) == pytest.approx(base - np.log(2))
    assert lprob([0, 0, 0]) == pytest.approx(base - np.log(4))
    assert lprob([0, 0, 1]) == pytest.approx(base - np.log(4))

    # ----------- Fail on second state ---------------------
    base = np.log(4) - np.log(17)
    assert lprob([1, 0]) == pytest.approx(base)
    assert lprob([1, 0, 0]) == pytest.approx(base - np.log(2))
    assert lprob([1, 0, 1]) == pytest.approx(base - np.log(2))
    assert lprob([1, 0, 0, 0]) == pytest.approx(base - np.log(4))
    assert lprob([1, 1, 0, 1]) == pytest.approx(base - np.log(4))

    with pytest.raises(ValueError):
        lprob([1, 1, 1, 1, 1, 1])

    example = list(actor.sample())
    assert -float('inf') < lprob(example)

    ctrl = actor.policy()
    example = []
    for env in [None, 0, 0, 0]:
        example.append(ctrl.send(env))
    assert -float('inf') < lprob(example)

    actor = fit(model, 0.7)
    assert actor.sat_prob() == pytest.approx(0.7)

 
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
    actor = improviser(model, coeff)
    
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

    vals = sorted(list(actor.node2val.values()))
    assert all(x == y for x, y in zip(vals, expected))

    #assert actor.prob([]) == 1
