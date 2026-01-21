# type-tests for `signal/_peak_finding.pyi`

# ruff: noqa: ERA001

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import argrelextrema, argrelmax, argrelmin, find_peaks, find_peaks_cwt, peak_prominences, peak_widths

###

_Int1D: TypeAlias = onp.Array1D[np.int_]
_Float1D: TypeAlias = onp.Array1D[np.float64]

###

_i64_1d: onp.Array1D[np.int_]
_f64_1d: onp.Array1D[np.float64]

###

# argrelmin

assert_type(argrelmin(_i64_1d), tuple[onp.ArrayND[np.intp], ...])
assert_type(argrelmin(_f64_1d), tuple[onp.ArrayND[np.intp], ...])

# argrelmax

assert_type(argrelmax(_i64_1d), tuple[onp.ArrayND[np.intp], ...])
assert_type(argrelmax(_f64_1d), tuple[onp.ArrayND[np.intp], ...])

# argrelextrema

def _cmp_i64(a: onp.ArrayND[np.int_], b: onp.ArrayND[np.int_]) -> onp.ArrayND[np.bool_]: ...
def _cmp_f64(a: onp.ArrayND[np.float64], b: onp.ArrayND[np.float64]) -> onp.ArrayND[np.bool_]: ...

assert_type(argrelextrema(_i64_1d, _cmp_i64), tuple[onp.ArrayND[np.intp], ...])
assert_type(argrelextrema(_f64_1d, _cmp_f64), tuple[onp.ArrayND[np.intp], ...])

# peak_prominences

assert_type(peak_prominences(_f64_1d, _i64_1d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp], onp.ArrayND[np.intp]])

# peak_widths

assert_type(
    peak_widths(_f64_1d, _i64_1d),
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int_], onp.ArrayND[np.float64], onp.ArrayND[np.float64]],
)

# find_peaks_cwt

assert_type(find_peaks_cwt(_i64_1d, 0.5), onp.Array1D[np.int_])
assert_type(find_peaks_cwt(_f64_1d, 0.5), onp.Array1D[np.int_])

# find_peaks

# {}
assert_type(find_peaks(_f64_1d)[0], _Int1D)
_, props_0 = find_peaks(_f64_1d)
props_0["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_0["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_0["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_0["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_0["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height}
_, props_h = find_peaks(_f64_1d, height=_f64_1d)
assert_type(props_h["peak_heights"], _Float1D)
props_h["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_h["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_h["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_h["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {threshold}
_, props_t = find_peaks(_f64_1d, threshold=_f64_1d)
props_t["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_t["left_thresholds"], _Float1D)
props_t["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_t["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_t["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {prominence}
_, props_p = find_peaks(_f64_1d, prominence=_f64_1d)
props_p["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_p["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_p["prominences"], _Float1D)
props_p["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_p["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {width}
_, props_w = find_peaks(_f64_1d, width=_f64_1d)
props_w["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_w["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_w["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_w["widths"], _Float1D)
props_w["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {plateau_size}
_, props_s = find_peaks(_f64_1d, plateau_size=_f64_1d)
props_s["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_s["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_s["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_s["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_s["plateau_sizes"], _Int1D)

# {height, threshold}
_, props_ht = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d)
assert_type(props_ht["peak_heights"], _Float1D)
assert_type(props_ht["left_thresholds"], _Float1D)
props_ht["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_ht["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_ht["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, prominence}
_, props_hp = find_peaks(_f64_1d, height=_f64_1d, prominence=_f64_1d)
assert_type(props_hp["peak_heights"], _Float1D)
props_hp["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hp["prominences"], _Float1D)
props_hp["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_hp["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, width}
_, props_hw = find_peaks(_f64_1d, height=_f64_1d, width=_f64_1d)
assert_type(props_hw["peak_heights"], _Float1D)
props_hw["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_hw["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hw["widths"], _Float1D)
props_hw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, plateau_size}
_, props_hs = find_peaks(_f64_1d, height=_f64_1d, plateau_size=_f64_1d)
assert_type(props_hs["peak_heights"], _Float1D)
props_hs["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_hs["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_hs["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hs["plateau_sizes"], _Int1D)

# {threshold, prominence}
_, props_tp = find_peaks(_f64_1d, threshold=_f64_1d, prominence=_f64_1d)
props_tp["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tp["left_thresholds"], _Float1D)
assert_type(props_tp["prominences"], _Float1D)
props_tp["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_tp["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {threshold, width}
_, props_tw = find_peaks(_f64_1d, threshold=_f64_1d, width=_f64_1d)
props_tw["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tw["left_thresholds"], _Float1D)
props_tw["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tw["widths"], _Float1D)
props_tw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {threshold, plateau_size}
_, props_ts = find_peaks(_f64_1d, threshold=_f64_1d, plateau_size=_f64_1d)
props_ts["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_ts["left_thresholds"], _Float1D)
props_ts["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_ts["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_ts["plateau_sizes"], _Int1D)

# {prominence, width}
_, props_pw = find_peaks(_f64_1d, prominence=_f64_1d, width=_f64_1d)
props_pw["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_pw["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_pw["prominences"], _Float1D)
assert_type(props_pw["widths"], _Float1D)
props_pw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {prominence, plateau_size}
_, props_ps = find_peaks(_f64_1d, prominence=_f64_1d, plateau_size=_f64_1d)
props_ps["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_ps["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_ps["prominences"], _Float1D)
props_ps["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_ps["plateau_sizes"], _Int1D)

# {width, plateau_size}
_, props_ws = find_peaks(_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
props_ws["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_ws["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_ws["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_ws["widths"], _Float1D)
assert_type(props_ws["plateau_sizes"], _Int1D)

# {height, threshold, prominence}
_, props_htp = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, prominence=_f64_1d)
assert_type(props_htp["peak_heights"], _Float1D)
assert_type(props_htp["left_thresholds"], _Float1D)
assert_type(props_htp["prominences"], _Float1D)
props_htp["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_htp["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, threshold, width}
_, props_htw = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, width=_f64_1d)
assert_type(props_htw["peak_heights"], _Float1D)
assert_type(props_htw["left_thresholds"], _Float1D)
props_htw["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_htw["widths"], _Float1D)
props_htw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, threshold, plateau_size}
_, props_hts = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, plateau_size=_f64_1d)
assert_type(props_hts["peak_heights"], _Float1D)
assert_type(props_hts["left_thresholds"], _Float1D)
props_hts["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_hts["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hts["plateau_sizes"], _Int1D)

# {height, prominence, width}
_, props_hpw = find_peaks(_f64_1d, height=_f64_1d, prominence=_f64_1d, width=_f64_1d)
assert_type(props_hpw["peak_heights"], _Float1D)
props_hpw["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hpw["prominences"], _Float1D)
assert_type(props_hpw["widths"], _Float1D)
props_hpw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, prominence, plateau_size}
_, props_hps = find_peaks(_f64_1d, height=_f64_1d, prominence=_f64_1d, plateau_size=_f64_1d)
assert_type(props_hps["peak_heights"], _Float1D)
props_hps["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hps["prominences"], _Float1D)
props_hps["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hps["plateau_sizes"], _Int1D)

# {height, width, plateau_size}
_, props_hws = find_peaks(_f64_1d, height=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
assert_type(props_hws["peak_heights"], _Float1D)
props_hws["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_hws["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hws["widths"], _Float1D)
assert_type(props_hws["plateau_sizes"], _Int1D)

# {threshold, prominence, width}
_, props_tpw = find_peaks(_f64_1d, threshold=_f64_1d, prominence=_f64_1d, width=_f64_1d)
props_tpw["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tpw["left_thresholds"], _Float1D)
assert_type(props_tpw["prominences"], _Float1D)
assert_type(props_tpw["widths"], _Float1D)
props_tpw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {threshold, prominence, plateau_size}
_, props_tps = find_peaks(_f64_1d, threshold=_f64_1d, prominence=_f64_1d, plateau_size=_f64_1d)
props_tps["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tps["left_thresholds"], _Float1D)
assert_type(props_tps["prominences"], _Float1D)
props_tps["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tps["plateau_sizes"], _Int1D)

# {threshold, width, plateau_size}
_, props_tws = find_peaks(_f64_1d, threshold=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
props_tws["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tws["left_thresholds"], _Float1D)
props_tws["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tws["widths"], _Float1D)
assert_type(props_tws["plateau_sizes"], _Int1D)

# {prominence, width, plateau_size}
_, props_pws = find_peaks(_f64_1d, prominence=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
props_pws["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
props_pws["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_pws["prominences"], _Float1D)
assert_type(props_pws["widths"], _Float1D)
assert_type(props_pws["plateau_sizes"], _Int1D)

# {height, threshold, prominence, width}
_, props_htpw = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, prominence=_f64_1d, width=_f64_1d)
assert_type(props_htpw["peak_heights"], _Float1D)
assert_type(props_htpw["left_thresholds"], _Float1D)
assert_type(props_htpw["prominences"], _Float1D)
assert_type(props_htpw["widths"], _Float1D)
props_htpw["plateau_sizes"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]

# {height, threshold, prominence, plateau_size}
_, props_htps = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, prominence=_f64_1d, plateau_size=_f64_1d)
assert_type(props_htps["peak_heights"], _Float1D)
assert_type(props_htps["left_thresholds"], _Float1D)
assert_type(props_htps["prominences"], _Float1D)
props_htps["widths"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_htps["plateau_sizes"], _Int1D)

# {height, threshold, width, plateau_size}
_, props_htws = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
assert_type(props_htws["peak_heights"], _Float1D)
assert_type(props_htws["left_thresholds"], _Float1D)
props_htws["prominences"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_htws["widths"], _Float1D)
assert_type(props_htws["plateau_sizes"], _Int1D)

# {height, prominence, width, plateau_size}
_, props_hpws = find_peaks(_f64_1d, height=_f64_1d, prominence=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
assert_type(props_hpws["peak_heights"], _Float1D)
props_hpws["left_thresholds"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_hpws["prominences"], _Float1D)
assert_type(props_hpws["widths"], _Float1D)
assert_type(props_hpws["plateau_sizes"], _Int1D)

# {threshold, prominence, width, plateau_size}
_, props_tpws = find_peaks(_f64_1d, threshold=_f64_1d, prominence=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
props_tpws["peak_heights"]  # type: ignore[typeddict-item]  # pyright: ignore[reportGeneralTypeIssues]  # pyrefly: ignore[bad-typed-dict-key]
assert_type(props_tpws["left_thresholds"], _Float1D)
assert_type(props_tpws["prominences"], _Float1D)
assert_type(props_tpws["widths"], _Float1D)
assert_type(props_tpws["plateau_sizes"], _Int1D)

# {height, threshold, prominence, width, plateau_size}
_, props_htpws = find_peaks(_f64_1d, height=_f64_1d, threshold=_f64_1d, prominence=_f64_1d, width=_f64_1d, plateau_size=_f64_1d)
assert_type(props_htpws["peak_heights"], _Float1D)
assert_type(props_htpws["left_thresholds"], _Float1D)
assert_type(props_htpws["prominences"], _Float1D)
assert_type(props_htpws["widths"], _Float1D)
assert_type(props_htpws["plateau_sizes"], _Int1D)
