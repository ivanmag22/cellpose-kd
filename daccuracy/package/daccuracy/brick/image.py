"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import sys as sstm
import typing as h

import numpy as nmpy
import scipy.ndimage as spim

array_t = nmpy.ndarray


def DetectionWithTolerance(detection: array_t, tolerance: float, /) -> array_t:
    """"""
    if tolerance < 1.0:
        return detection

    output = nmpy.zeros_like(detection)

    distance_map = spim.distance_transform_edt(detection != 1)
    output[distance_map <= tolerance] = 1

    for label in range(2, nmpy.amax(detection).item() + 1):
        current_map = spim.distance_transform_edt(detection != label)
        closer_bmap = current_map < distance_map
        output[nmpy.logical_and(closer_bmap, current_map <= tolerance)] = label
        distance_map[closer_bmap] = current_map[closer_bmap]

    return output


def ShiftedVersion(image: array_t, shifts: h.Sequence[int], /) -> array_t:
    """"""
    if shifts.__len__() != image.ndim:
        print(
            f"{shifts}/{image.ndim}: Incompatible requested shifts and image dimension. Using image as-is.",
            file=sstm.stderr,
        )
        return image

    output = image

    everything = slice(None)
    for d_idx in range(image.ndim):
        if (shift := shifts[d_idx]) != 0:
            output = nmpy.roll(output, shift, axis=d_idx)
            if shift > 0:
                slice_ = slice(shift)
            else:
                slice_ = slice(shift, None)
            slices = (
                (d_idx * (everything,))
                + (slice_,)
                + ((image.ndim - d_idx - 1) * (everything,))
            )
            output[slices] = 0

    return output


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
