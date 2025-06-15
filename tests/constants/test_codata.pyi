from typing import assert_type

from scipy import constants

assert_type(constants.value("elementary charge"), float)
assert_type(type(constants.unit("proton mass")), type[str])
assert_type(constants.precision("proton mass"), float)

assert_type(constants.find("boltzmann"), list[str])
assert_type(constants.find("boltzmann", False), list[str])
assert_type(constants.find("boltzmann", disp=False), list[str])
assert_type(constants.find("boltzmann", True), None)
assert_type(constants.find("boltzmann", disp=True), None)

assert_type(constants.physical_constants["classical electron radius"][0], float)
assert_type(type(constants.physical_constants["classical electron radius"][1]), type[str])
assert_type(constants.physical_constants["classical electron radius"][2], float)
