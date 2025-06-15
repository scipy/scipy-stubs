from typing import LiteralString, assert_type

from scipy import constants

assert_type(constants.value("elementary charge"), float)
constants.value(None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
constants.value(b"proton mass")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

assert_type(constants.unit("proton mass"), LiteralString)
constants.unit(None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
constants.unit(b"proton mass")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

assert_type(constants.precision("proton mass"), float)
constants.precision(None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
constants.precision(b"proton mass")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

assert_type(constants.find(None), list[str])
assert_type(constants.find("boltzmann"), list[str])
assert_type(constants.find("boltzmann", False), list[str])
assert_type(constants.find("boltzmann", disp=False), list[str])
assert_type(constants.find("boltzmann", True), None)
assert_type(constants.find("boltzmann", disp=True), None)
constants.find(b"boltzmann")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]

assert_type(constants.physical_constants, dict[str, tuple[float, str, float]])
