import aiger_bv as BV
import aiger_coins as C

from aiger_improv.model import from_pcirc


def test_from_pcirc_smoke():
    sys = BV.uatom(3, 'sys')
    env = BV.uatom(3, 'env')
    out = BV.uatom(3, 'out')

    monitor = out < 4

    dyn = (sys + env).with_output('out')
    dyn = C.pcirc(dyn) \
           .randomize({'env': {0: 1/3, 1: 1/3, 5: 1/3}}) \
           .with_coins_id('coins')

    model = from_pcirc(dyn, monitor, steps=2)
    assert model.coin_biases('sys##time_0') is None

    coins0 = f'{dyn.coins_id}##time_0'
    assert model.coin_biases(coins0) == dyn.coin_biases

    assert model.mdd({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 0,
        'coins##time_1': 0b11,  # <- env == 0
    }) == 1

    assert model.mdd({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 5,
        'coins##time_1': 0b11,  # <- env == 0
    }) == 0

    sum_zero0 = BV.uatom(3, f'out##time_1') == 0b000
    expr2 = model.preimage(sum_zero0)

    assert expr2({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 0,
        'coins##time_1': 0b11,  # <- env == 0
    })[0]

    assert expr2({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 0,
        'coins##time_1': 0b00,  # <- env != 0
    })[0]

    assert not expr2({
        'sys##time_0': 0,
        'coins##time_0': 0b00,  # <- env != 0
        'sys##time_1': 0,
        'coins##time_1': 0b00,  # <- env != 0
    })[0]

    model2 = model.override(sum_zero0, False)

    assert model2.mdd({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 0,
        'coins##time_1': 0b11,  # <- env == 0
    }) == 0

    assert model2.mdd({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 3,
        'coins##time_1': 0b11,  # <- env == 0
    }) == 0

    assert model2.mdd({
        'sys##time_0': 0,
        'coins##time_0': 0b11,  # <- env == 0
        'sys##time_1': 5,
        'coins##time_1': 0b11,  # <- env == 0
    }) == 0

    graph = model.graph()
    graph2 = model2.graph()
    assert len(graph2) >= len(graph)  # Now need to case split at 0.

    assert not model.is_coin('sys##time_0')
    assert model.is_coin('coins##time_0')
