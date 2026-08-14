"""Unit tests for x-opm transforms & classification.

Runnable with pytest OR directly: ``python x-opm/tests/test_x_opm.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import TypeRules, classify, parse_column, should_invert
from transform import (scale_hamming, scale_raw_int, split_bits, toggle_hamming)

R = TypeRules()


def _obj(*xs):
    return np.array(list(xs), dtype=object)


def test_parse_column():
    assert parse_column("a/b[7:0]") == ("a/b", 8, 0)
    assert parse_column("a/b[14:2]") == ("a/b", 13, 2)
    assert parse_column("a/b") == ("a/b", 1, 0)
    # single index (instance array) has no colon -> kept verbatim, width 1
    assert parse_column("a/RAM[0]/clk") == ("a/RAM[0]/clk", 1, 0)


def test_classify_priority():
    assert classify("x/regs_clk_en", R)[0] == "B"          # clk + _en
    assert classify("x/x_regs_clk/global_en", R)[0] == "B"  # clk via path + _en
    assert classify("x/fence_clk", R)[0] == "D"            # 'en' from 'fence' -> NOT B
    assert classify("x/cp0_idu_issue_stall", R)[0] == "A"  # _stall
    assert classify("x/rd_data_vld", R)[0] == "A"          # _vld beats data
    assert classify("x/mem_wdata[7:0]", R)[0] == "C"       # data bus
    assert classify("x/epc[63:0]", R)[0] == "D"            # residual payload
    assert should_invert("x/cp0_idu_issue_stall", R) is True
    assert should_invert("x/foo_req", R) is False


def test_split_bits_basic():
    # values 0b1011=11, 0b0100=4 ; width 4, lo 0
    out = split_bits(_obj(11, 4), "s", 4, 0)
    names = [n for n, _ in out]
    assert names == ["s[0]", "s[1]", "s[2]", "s[3]"]
    got = {n: a.tolist() for n, a in out}
    assert got["s[0]"] == [1, 0]
    assert got["s[1]"] == [1, 0]
    assert got["s[2]"] == [0, 1]
    assert got["s[3]"] == [1, 0]


def test_split_bits_lo_offset_and_invert():
    # value bit index uses k (shift by k); label uses lo+k
    out = split_bits(_obj(11), "s", 4, 2, invert=True)
    got = {n: a.tolist() for n, a in out}
    assert set(got) == {"s[2]", "s[3]", "s[4]", "s[5]"}
    # 11 = 0b1011 -> bit0=1 bit1=1 bit2=0 bit3=1 ; inverted
    assert got["s[2]"] == [0]
    assert got["s[3]"] == [0]
    assert got["s[4]"] == [1]
    assert got["s[5]"] == [0]


def test_split_bits_width1_keeps_name():
    out = split_bits(_obj(1, 0), "s/sig", 1, 0)
    assert [n for n, _ in out] == ["s/sig"]


def test_split_bits_wide():
    out = split_bits(_obj((1 << 70) + 1), "s", 71, 0)
    got = {n: a.tolist() for n, a in out}
    assert got["s[0]"] == [1]
    assert got["s[70]"] == [1]
    assert got["s[69]"] == [0]


def test_toggle_hamming():
    vals = _obj(0b0011, 0b0110, 0b0110)
    ham = toggle_hamming(vals, 4, count_initial=True)
    assert ham.tolist() == [2, 2, 0]           # pc(3), pc(3^6=5), pc(0)
    ham0 = toggle_hamming(vals, 4, count_initial=False)
    assert ham0.tolist() == [0, 2, 0]


def test_toggle_hamming_wide():
    vals = _obj(1 << 70, 0)
    ham = toggle_hamming(vals, 71, count_initial=True)
    assert ham.tolist() == [1, 1]


def test_scaling():
    assert scale_raw_int(_obj(0, 255), 8).tolist() == [0.0, 1.0]
    assert scale_hamming(np.array([0, 2, 4], dtype=np.int32), 4).tolist() == [0.0, 0.5, 1.0]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
