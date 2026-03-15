"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import typing as h

import daccuracy.brick.image as imge
import numpy as nmpy
import scipy.optimize as spop
import skimage.segmentation as sisg

_MEASURE_COUNTS_HEADER = ("n ground truths", "n detections")


array_t = nmpy.ndarray


class pointwise_measures_t(h.NamedTuple):
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1_score: float
    froc_sample: tuple[int, float]
    check_tp_fn_equal_gt: int
    check_tp_fp_equal_dn: int
    dn_2_gt_associations: dict[int, int]


class region_measures_t(h.NamedTuple):
    overlap_mean: float
    overlap_stddev: float
    overlap_min: float
    overlap_max: float
    jaccard_mean: float
    jaccard_stddev: float
    jaccard_min: float
    jaccard_max: float
    precision_p_mean: float
    precision_p_stddev: float
    precision_p_min: float
    precision_p_max: float
    recall_p_mean: float
    recall_p_stddev: float
    recall_p_min: float
    recall_p_max: float
    f1_score_p_mean: float
    f1_score_p_stddev: float
    f1_score_p_min: float
    f1_score_p_max: float


full_pointwise_measures_h = tuple[int, int, pointwise_measures_t]
full_pw_region_measures_h = tuple[int, int, pointwise_measures_t, region_measures_t]
full_measures_h = full_pointwise_measures_h | full_pw_region_measures_h

measure_fct_h = h.Callable[..., tuple[str, ...] | full_measures_h]


def AccuracyMeasures(
    ground_truth: array_t,
    detection: array_t,
    measure_fct: measure_fct_h,
    should_exclude_border: bool,
    tolerance: float,
    /,
) -> full_measures_h:
    """"""
    if should_exclude_border:
        sisg.clear_border(ground_truth, out=ground_truth)
        sisg.clear_border(detection, out=detection)
        ground_truth, *_ = sisg.relabel_sequential(ground_truth)
        detection, *_ = sisg.relabel_sequential(detection)

    return measure_fct(ground_truth, detection, tolerance=tolerance)


def MeasuresAsStrings(measures: full_measures_h, /) -> list[str]:
    """"""
    output = [elm.__str__() for elm in measures[:2]]  # Object counts

    for group in measures[2:]:
        output.extend(elm.__str__() for elm in group._asdict().values())

    return output


def PointwiseMeasures(
    ground_truth: array_t | None,
    detection: array_t | None,
    /,
    *,
    tolerance: float = 0.0,
) -> tuple[str, ...] | full_pointwise_measures_h:
    """"""
    if ground_truth is None:
        return *_MEASURE_COUNTS_HEADER, *pointwise_measures_t._fields

    if tolerance >= 1.0:
        detection = imge.DetectionWithTolerance(detection, tolerance)

    n_gt_objects = nmpy.amax(ground_truth).item()
    n_dn_objects = nmpy.amax(detection).item()

    dn_2_gt_associations = ObjectAssociations(
        n_dn_objects, detection, n_gt_objects, ground_truth
    )
    correct = dn_2_gt_associations.__len__()
    missed = n_gt_objects - correct
    invented = n_dn_objects - correct

    output = (
        n_gt_objects,
        n_dn_objects,
        _StandardMeasuresFromCounts(correct, missed, invented, dn_2_gt_associations),
    )

    return output


def PWandRegionMeasures(
    ground_truth: array_t | None,
    detection: array_t | None,
    /,
    *,
    tolerance: float = 0.0,
) -> tuple[str, ...] | full_pw_region_measures_h:
    """"""
    if ground_truth is None:
        return (
            *_MEASURE_COUNTS_HEADER,
            *pointwise_measures_t._fields,
            *region_measures_t._fields,
        )

    n_gt_objects, n_dn_objects, pointwise_measures = PointwiseMeasures(
        ground_truth, detection, tolerance=tolerance
    )

    overlap, jaccard, precision_p, recall_p, f1_score_p = [], [], [], [], []
    for dn_label, gt_label in pointwise_measures.dn_2_gt_associations.items():
        ground_truth_obj = ground_truth == gt_label
        detected_obj = detection == dn_label

        gt_area = nmpy.count_nonzero(ground_truth_obj)
        dn_area = nmpy.count_nonzero(detected_obj)
        union_area = nmpy.count_nonzero(nmpy.logical_or(ground_truth_obj, detected_obj))
        intersection_area = nmpy.count_nonzero(
            nmpy.logical_and(ground_truth_obj, detected_obj)
        )
        assert intersection_area > 0, "This should never happen; Contact Developer"
        one_precision = intersection_area / dn_area
        one_recall = intersection_area / gt_area

        overlap.append(100.0 * intersection_area / min(gt_area, dn_area))
        jaccard.append(intersection_area / union_area)
        precision_p.append(one_precision)
        recall_p.append(one_recall)
        f1_score_p.append(
            2.0 * one_precision * one_recall / (one_precision + one_recall)
        )

    measures = {}
    for name, values in zip(
        ("overlap", "jaccard", "precision_p", "recall_p", "f1_score_p"),
        (overlap, jaccard, precision_p, recall_p, f1_score_p),
    ):
        for reduction, as_numpy in zip(
            ("mean", "stddev", "min", "max"), ("mean", "std", "amin", "amax")
        ):
            ReductionFunction = getattr(nmpy, as_numpy)
            measures[f"{name}_{reduction}"] = ReductionFunction(values).item()

    region_measures = region_measures_t(**measures)

    return n_gt_objects, n_dn_objects, pointwise_measures, region_measures


def ObjectAssociations(
    n_ref_objects: int, ref_img: array_t, n_objects: int, image: array_t, /
) -> dict[int, int]:
    """"""
    assignment_costs = nmpy.ones((n_ref_objects, n_objects), dtype=nmpy.float64)

    for ref_label in range(1, n_ref_objects + 1):
        ref_obj = ref_img == ref_label

        corresponding_labels = nmpy.unique(image[ref_obj])
        if corresponding_labels[0] == 0:
            corresponding_labels = corresponding_labels[1:]

        for crr_label in corresponding_labels:
            crr_label = crr_label.item()
            corresponding_obj = image == crr_label

            union_area = nmpy.count_nonzero(nmpy.logical_or(corresponding_obj, ref_obj))
            intersection_area = nmpy.count_nonzero(
                nmpy.logical_and(corresponding_obj, ref_obj)
            )

            assignment_costs[ref_label - 1, crr_label - 1] = (
                1.0 - intersection_area / union_area
            )

    row_ind, col_ind = spop.linear_sum_assignment(assignment_costs)
    valid_idc = assignment_costs[row_ind, col_ind] < 1.0
    output = dict(zip(row_ind[valid_idc] + 1, col_ind[valid_idc] + 1))

    return output


def _StandardMeasuresFromCounts(
    correct: int, missed: int, invented: int, dn_2_gt_associations: dict[int, int], /
) -> pointwise_measures_t:
    """"""
    true_positive = correct
    false_positive = invented
    false_negative = missed

    if true_positive > 0:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    else:
        precision = 0.0
        recall = 0.0
    if (precision > 0.0) and (recall > 0.0):
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    true_positive_rate = recall
    froc_sample = (false_positive, true_positive_rate)

    output = pointwise_measures_t(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        froc_sample=froc_sample,
        check_tp_fn_equal_gt=correct + missed,
        check_tp_fp_equal_dn=correct + invented,
        dn_2_gt_associations=dn_2_gt_associations,
    )

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
