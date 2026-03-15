"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import typing as h

row_transform_h = h.Callable[[float], float]


COL_SEPARATOR = "; "


# --- INPUT


def SymmetrizedRow(idx: float, img_height: float, /) -> float:
    """"""
    return img_height - idx - 1.0


def ColLabelToIdx(label: str, /) -> int:
    """"""
    reference = ord("A")
    if (length := label.__len__()) > 1:
        ords = map(lambda _ltt: ord(_ltt) - reference + 1, reversed(label))
        powers = (26**_idx for _idx in range(length))
        output = sum(_ord * _pwr for _ord, _pwr in zip(ords, powers)) - 1
    else:
        output = ord(label) - reference

    return output


def CSVLineToCoords(
    # line instead of row to avoid confusion with row index of center
    line: h.Sequence[str],
    coordinate_idc: h.Sequence[int] | None,
    row_transform: row_transform_h,
    /,
) -> tuple[int, ...] | None:
    """"""
    if coordinate_idc is None:
        coordinate_idc = tuple(range(line.__len__()))

    try:
        row = float(line[coordinate_idc[0]])
    except ValueError:
        # CSV header line
        return None

    row = row_transform(row)
    remaining = [float(line[_idx]) for _idx in coordinate_idc[1:]]

    return tuple(int(round(_elm)) for _elm in [row] + remaining)


# --- OUTPUT


def HeaderRow(measure_header: h.Sequence[str], /) -> h.Sequence[str]:
    """"""
    measure_header = tuple(_elm.capitalize() for _elm in measure_header)

    return "Ground truth", "Detection", *measure_header


"""
COPYRIGHT NOTICE

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

SEE LICENCE NOTICE: file README-LICENCE-utf8.txt at project source root.

This software is being developed by Eric Debreuve, a CNRS employee and
member of team Morpheme.
Team Morpheme is a joint team between Inria, CNRS, and UniCA.
It is hosted by the Centre Inria d'Université Côte d'Azur, Laboratory
I3S, and Laboratory iBV.

CNRS: https://www.cnrs.fr/index.php/en
Inria: https://www.inria.fr/en/
UniCA: https://univ-cotedazur.eu/
Centre Inria d'Université Côte d'Azur: https://www.inria.fr/en/centre/sophia/
I3S: https://www.i3s.unice.fr/en/
iBV: http://ibv.unice.fr/
Team Morpheme: https://team.inria.fr/morpheme/
"""
