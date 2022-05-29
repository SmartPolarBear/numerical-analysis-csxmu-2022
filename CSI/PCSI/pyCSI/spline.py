from typing import Iterable

import numpy as np
from enum import Enum

from pyCSI.impl.derivative1 import spline_impl_derive1
from pyCSI.impl.derivative2 import spline_impl_derive2
from pyCSI.impl.periodic import spline_impl_periodic
from pyCSI.impl.notaknot import spline_impl_not_a_knot
from pyCSI.impl.utils import create_function


class ConstraintType(Enum):
    NOT_A_KNOT = 1
    NATURAL = 2,
    DERIVATIVE1 = 3
    DERIVATIVE2 = 4,
    PERIODIC = 5


def spline(x: Iterable, y: Iterable, constraint_type: ConstraintType = ConstraintType.NOT_A_KNOT,
           symbolic_result: bool = True, **constraints):
    x, y = np.array(x), np.array(y)
    h = np.diff(x)

    co1 = co2 = co3 = co4 = None
    if constraint_type == ConstraintType.DERIVATIVE1:
        co1, co2, co3, co4 = spline_impl_derive1(x, y, h, constraints["m0"], constraints["mn"])
    elif constraint_type == ConstraintType.DERIVATIVE2:
        co1, co2, co3, co4 = spline_impl_derive2(x, y, h, constraints["M0"], constraints["Mn"])
    elif constraint_type == ConstraintType.NATURAL:
        co1, co2, co3, co4 = spline_impl_derive2(x, y, h, M0=0, Mn=0)
    elif constraint_type == ConstraintType.PERIODIC:
        co1, co2, co3, co4 = spline_impl_periodic(x, y, h)
    elif constraint_type == ConstraintType.NOT_A_KNOT:
        co1, co2, co3, co4 = spline_impl_not_a_knot(x, y, h)
    else:
        raise RuntimeError("Invalid constraint type {}".format(constraint_type))

    if symbolic_result:
        return create_function(co1, co2, co3, co4, x, x.shape[0] - 1)
    else:
        return np.transpose(np.matrix([co1, co2, co3, co4]))
