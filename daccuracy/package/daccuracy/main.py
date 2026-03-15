"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

from pathlib import Path as path_t
from typing import Any, Dict, Optional, Sequence, Tuple

import daccuracy.brick.measures as msrs
import daccuracy.io.csv_ as csio
import daccuracy.io.input as inpt
import daccuracy.io.output as otpt
from daccuracy.brick.measures import measure_fct_h
from daccuracy.io.csv_ import row_transform_h


def ComputeAndOutputMeasures(options: Dict[str, Any], /) -> None:
    """"""
    gt_dn_mode: str = options["gt_dn_mode"]
    ground_truth_folder: Optional[path_t] = options["ground_truth_folder"]
    ground_truth_path: path_t = options["ground_truth_path"]
    relabel_gt: Optional[str] = options["relabel_gt"]
    detection_path: path_t = options["detection_path"]
    relabel_dn: Optional[str] = options["relabel_dn"]
    measure_fct: measure_fct_h = options["measure_fct"]
    coord_trans: Optional[Tuple[Sequence[int], row_transform_h]] = options[
        "coord_trans"
    ]
    dn_shifts: Optional[Sequence[int]] = options["dn_shifts"]
    should_exclude_border: bool = options["should_exclude_border"]
    tolerance: float = options["tolerance"]
    output_format: str = options["output_format"]
    should_show_image: bool = options["should_show_image"]
    output_accessor = options["output_accessor"]

    if detection_path.is_file():
        detection_folder = detection_path.parent
        detection_name = detection_path.name
    else:
        detection_folder = detection_path
        detection_name = None

    if gt_dn_mode == "one-to-many":
        coordinate_idc, row_transform = coord_trans
        header = csio.HeaderRow(measure_fct(None, None))
    else:
        coordinate_idc = row_transform = None
        header = csio.HeaderRow(msrs.PWandRegionMeasures(None, None))
    if output_format == "csv":
        print(csio.COL_SEPARATOR.join(header), file=output_accessor)
        name_field_len = 0
    else:
        name_field_len = max(elm.__len__() for elm in header)

    figures_are_waiting = False
    for document in detection_folder.iterdir():
        if document.is_file() and (
            (detection_name is None) or (document.name == detection_name)
        ):
            detection = inpt.ImageAtPath(
                document, relabel_dn, dn_shifts, None, None, None
            )
            if detection is None:
                continue

            ground_truth_path = inpt.GroundTruthPathForDetection(
                document.stem, ground_truth_path, ground_truth_folder, gt_dn_mode
            )
            if ground_truth_path is None:
                continue
            if gt_dn_mode == "one-to-one":
                if ground_truth_path.suffix.lower() == ".csv":
                    if coord_trans is None:
                        coordinate_idc = None
                        row_transform = lambda f_idx: f_idx
                    else:
                        coordinate_idc, row_transform = coord_trans
                    measure_fct = msrs.PointwiseMeasures
                else:
                    measure_fct = msrs.PWandRegionMeasures
            ground_truth = inpt.ImageAtPath(
                ground_truth_path,
                relabel_gt,
                None,
                detection.shape,
                coordinate_idc,
                row_transform,
            )
            if ground_truth is None:
                continue

            if (gt_shape := ground_truth.shape) != (dn_shape := detection.shape):
                if fixable := (sorted((ground_truth.ndim, detection.ndim)) == [2, 3]):
                    ground_truth, detection = inpt.WithFixedDimensions(
                        ground_truth, detection
                    )
                if (not fixable) or (ground_truth is None) or (detection is None):
                    print(
                        f"{gt_shape} != {dn_shape}: Ground-truth and detection shapes mismatch "
                        f"for images {ground_truth_path} and {document}"
                    )
                    continue

            measures = msrs.AccuracyMeasures(
                ground_truth, detection, measure_fct, should_exclude_border, tolerance
            )
            measures_as_str = msrs.MeasuresAsStrings(measures)
            output_row = [ground_truth_path.name, document.name] + measures_as_str

            if output_format == "csv":
                print(csio.COL_SEPARATOR.join(output_row), file=output_accessor)
            else:
                for name, value in zip(header, output_row):
                    print(f"{name:>{name_field_len}} = {value}", file=output_accessor)

            if should_show_image:
                """
                print(
                    f"GT:\t{ground_truth.ndim}, {ground_truth.shape}",
                    file=output_accessor,
                )
                """
                if ground_truth.ndim == 2:
                    otpt.PrepareMixedGTDetectionImage(
                        ground_truth,
                        detection,
                        dn_2_gt_associations=measures[2].dn_2_gt_associations,
                    )
                    if ground_truth_path.suffix.lower() != ".csv":
                        otpt.PrepareMixedGTDetectionImage(
                            ground_truth, detection, mode="pixel"
                        )
                elif ground_truth.ndim == 3:
                    assert ground_truth.shape == detection.shape
                    for i in range(ground_truth.shape[0]):  # gt.shape[0] = slice
                        otpt.PrepareMixedGTDetectionImage(
                            ground_truth[i],
                            detection[i],
                            dn_2_gt_associations=measures[2].dn_2_gt_associations,
                        )
                        if ground_truth_path.suffix.lower() != ".csv":
                            otpt.PrepareMixedGTDetectionImage(
                                ground_truth[i], detection[i], mode="pixel"
                            )
                        otpt.ShowPreparedImages()
                figures_are_waiting = True
            """
            if should_show_image and (ground_truth.ndim == 2):
                otpt.PrepareMixedGTDetectionImage(
                    ground_truth,
                    detection,
                    dn_2_gt_associations=measures[2].dn_2_gt_associations,
                )
                if ground_truth_path.suffix.lower() != ".csv":
                    otpt.PrepareMixedGTDetectionImage(
                        ground_truth, detection, mode="pixel"
                    )
                figures_are_waiting = True
            """

    if figures_are_waiting:
        otpt.ShowPreparedImages()


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
