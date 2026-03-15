"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

# Run from project folder with: python -m daccuracy.test.test-random

import sys as sstm
from pathlib import Path as path_t

import daccuracy.brick.measures as msre
import daccuracy.io.input as inpt
import daccuracy.io.output as oupt
import matplotlib.pyplot as pypl
import numpy as nmpy

# --- PARAMETERS
#
width = 120
height = 100
min_ground_truths = 10
max_ground_truths = 100
row_shift = 2
col_shift = 1

# --- RANDOM GROUND-TRUTH
#
gt_interval = (min_ground_truths, max_ground_truths + 1)
n_ground_truths = nmpy.random.randint(*gt_interval, size=1).item()

in_interval = (0, int(nmpy.around(0.3 * max_ground_truths).item()) + 1)
correct_ = nmpy.random.randint(0, n_ground_truths + 1, size=1).item()
missed_ = n_ground_truths - correct_
invented_ = nmpy.random.randint(*in_interval, size=1).item()

# - shifts to avoid rollbacks in the shifted detection below
gt_rows = nmpy.random.randint(0, height - row_shift, size=n_ground_truths)
gt_cols = nmpy.random.randint(0, width - col_shift, size=n_ground_truths)

ground_truth_ = nmpy.zeros((height, width), dtype=nmpy.uint8)
ground_truth_[(gt_rows, gt_cols)] = range(1, gt_rows.__len__() + 1)
if not inpt.LabeledImageIsValid(ground_truth_, path_t("ground-truth.synthetic"))[0]:
    print("Invalid random ground-truth image; Please re-try")
    sstm.exit(0)

# --- GROUND-TRUTH-INSPIRED DETECTION
#
detection_ = nmpy.zeros_like(ground_truth_)
#
correct_idc = nmpy.random.permutation(n_ground_truths)[:correct_]
detection_[(gt_rows[correct_idc], gt_cols[correct_idc])] = range(
    1, correct_idc.__len__() + 1
)
#
rows = nmpy.array(
    list(set(range(height)) - set(gt_rows)), dtype=int
)  # Too restrictive in such an independent way...
cols = nmpy.array(list(set(range(width)) - set(gt_cols)), dtype=int)  # ...but easier
common_length = min(rows.__len__(), cols.__len__())
rows = rows[:common_length]
cols = cols[:common_length]
if invented_ > common_length:
    invented_ = common_length
invented_row_idc = nmpy.random.permutation(common_length)[:invented_]
invented_col_idc = nmpy.random.permutation(common_length)[:invented_]
detection_[(rows[invented_row_idc], cols[invented_col_idc])] = (
    nmpy.arange(1, invented_col_idc.__len__() + 1) + correct_idc.__len__()
)
if not inpt.LabeledImageIsValid(detection_, path_t("detection.synthetic"))[0]:
    print("Invalid random detection image; Please re-try")
    sstm.exit(0)

pypl.matshow(detection_, cmap="Dark2_r")
pypl.gca().set_title(f"Detection (max={nmpy.amax(detection_)})")

# --- GROUND-TRUTH FEEDBACK
#
print(
    f"++++++++++\n"
    f"n_ground_truths = {n_ground_truths}\n"
    f"   n_detections = {nmpy.amax(detection_)}\n"
    f"        correct = {correct_}\n"
    f"         missed = {missed_}\n"
    f"       invented = {invented_}\n\n"
    f"/!\\ Verification can be made only if there is no warning about invalid ground-truth /!\\\n\n"
    f"++++++++++"
)
pypl.matshow(ground_truth_, cmap="Dark2_r")
pypl.gca().set_title(f"Ground truth (max={nmpy.amax(ground_truth_)})")

# --- OUTPUT PREPARATION
#
header = msre.PointwiseMeasures(None, None)
name_field_len = max(elm.__len__() for elm in header)

# --- PERFORMANCE MEASURES OUTPUT (ground-truth-inspired detection)
#
measures_ = msre.PointwiseMeasures(ground_truth_, detection_)
measures_as_str = msre.MeasuresAsStrings(measures_)
print(f"\n----------\nOriginal detection w/o tolerance\n----------")
for name, value in zip(header, measures_as_str):
    print(f"{name:>{name_field_len}} = {value}")
pypl.figure()
pypl.gca().set_title("Ground truth+Detection")
oupt.PrepareMixedGTDetectionImage(
    ground_truth_, detection_, dn_2_gt_associations=measures_[2].dn_2_gt_associations
)

# --- GROUND-TRUTH-SHIFTED DETECTION
#
shifted_detection = nmpy.roll(ground_truth_, (row_shift, col_shift), axis=(0, 1))
# Passing a fake path with a CSV extension disables labeling checking.
if not inpt.LabeledImageIsValid(shifted_detection, path_t("shifted_detection.csv"))[0]:
    print("Invalid random shifted detection image; Please re-try")
    sstm.exit(0)
assert nmpy.amax(shifted_detection) == nmpy.amax(ground_truth_)

pypl.matshow(shifted_detection, cmap="Dark2_r")
pypl.gca().set_title(
    f"Detection=Shifted Ground truth (max={nmpy.amax(shifted_detection)}=?={nmpy.amax(ground_truth_)})"
)

# --- PERFORMANCE MEASURES COMPUTATION & OUTPUT (ground-truth-shifted detection)
#
for tolerance_ in range(1, 6):
    measures_ = msre.PointwiseMeasures(
        ground_truth_, shifted_detection, tolerance=tolerance_
    )
    measures_as_str = msre.MeasuresAsStrings(measures_)
    print(
        f"\n"
        f"----------\n"
        f"Shifted detection w/ tolerance {tolerance_}\n"
        f"----------"
    )
    for name, value in zip(header, measures_as_str):
        print(f"{name:>{name_field_len}} = {value}")

    # shifted_detection_w_tol = imge.DetectionWithTolerance(shifted_detection, tolerance_)
    pypl.figure()
    pypl.gca().set_title(f"Ground truth+Shifted Detection with tol {tolerance_}")
    oupt.PrepareMixedGTDetectionImage(
        ground_truth_,
        shifted_detection,
        dn_2_gt_associations=measures_[2].dn_2_gt_associations,
    )

pypl.show()


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
