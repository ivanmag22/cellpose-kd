"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import re as rgex
import sys as sstm
import typing as h
from argparse import ArgumentParser as argument_parser_t
from argparse import RawDescriptionHelpFormatter
from pathlib import Path as path_t

# import __main__ as main_package
import daccuracy.brick.measures as msrs
import daccuracy.io.csv_ as csio
import daccuracy.io.output as otpt
from daccuracy.constant.app import DESCRIPTION
from daccuracy.constant.measures import MEASURES_DESCRIPTIONS
from daccuracy.io.csv_ import row_transform_h
from daccuracy.version import __version__


def ProcessedArguments(arguments: h.Sequence[str], /) -> dict[str, h.Any]:
    """"""
    parser = _ArgumentParser()
    # additional_args: optional rc or xy mode and corresponding columns in CSV
    known_options, additional_args = parser.parse_known_args(arguments)
    if (known_options.ground_truth_path is None) or (
        known_options.detection_path is None
    ):
        if known_options.should_explain_measures:
            print(MEASURES_DESCRIPTIONS)
            sstm.exit(0)
        else:
            parser.error("the following arguments are required: --gt, --dn")

    if (n_additional_args := additional_args.__len__()) > 1:
        print("Too many arguments", file=sstm.stderr)
        parser.print_help()
        sstm.exit(-1)

    ground_truth_path = path_t(known_options.ground_truth_path)
    detection_path = path_t(known_options.detection_path)
    if not (ground_truth_path.is_file() or ground_truth_path.is_dir()):
        print(f"{ground_truth_path}: Not a file or folder", file=sstm.stderr)
        sstm.exit(-1)
    if not (detection_path.is_file() or detection_path.is_dir()):
        print(f"{detection_path}: Not a file or folder", file=sstm.stderr)
        sstm.exit(-1)
    if detection_path.is_file() and not ground_truth_path.is_file():
        print(
            f"{ground_truth_path}: Not a file while detection is a file",
            file=sstm.stderr,
        )
        sstm.exit(-1)

    if ground_truth_path.is_file():
        gt_dn_mode = "one-to-many"
        ground_truth_folder = None
    else:
        gt_dn_mode = "one-to-one"
        ground_truth_folder = ground_truth_path

    if n_additional_args > 0:
        # coord_trans = coordinate_idc, row_transform
        coord_trans = _CoordinateIndices(additional_args[0], parser)
    else:
        coord_trans = None

    if gt_dn_mode == "one-to-many":
        if ground_truth_path.suffix.lower() == ".csv":
            if coord_trans is None:
                coord_trans = None, lambda f_idx: f_idx
            measure_fct = msrs.PointwiseMeasures
        else:
            coord_trans = None, None  # Just for convenience to call ImageAtPath
            measure_fct = msrs.PWandRegionMeasures
    else:
        measure_fct = None

    output = vars(known_options)
    output["gt_dn_mode"] = gt_dn_mode
    output["ground_truth_folder"] = ground_truth_folder
    output["ground_truth_path"] = ground_truth_path
    output["detection_path"] = detection_path
    output["coord_trans"] = coord_trans
    output["measure_fct"] = measure_fct

    return output


def _CoordinateIndices(
    option: str, parser: argument_parser_t, /
) -> tuple[h.Sequence[int], row_transform_h | None]:
    """"""
    match = rgex.match("--([rx])([A-Z]+)([cy])([A-Z]+)", option)

    if match is None:
        print(f"{option}: Invalid option", file=sstm.stderr)
        parser.print_help()
        sstm.exit(-1)

    if match.group(1) == "r":
        if match.group(3) != "c":
            print(f'{option}: "r"/"y" mixing', file=sstm.stderr)
            sstm.exit(-1)

        row_idx = csio.ColLabelToIdx(match.group(2))
        col_idx = csio.ColLabelToIdx(match.group(4))
        row_transform = lambda f_idx: f_idx
    #
    else:
        if match.group(3) != "y":
            print(f'{option}: "x"/"c" mixing', file=sstm.stderr)
            sstm.exit(-1)

        row_idx = csio.ColLabelToIdx(match.group(4))
        col_idx = csio.ColLabelToIdx(match.group(2))
        # This will later be changed into a symmetrization transform for each image
        row_transform = None

    remaining = option[match.end() :]
    matches = rgex.findall(r"\+[A-Z]+", remaining)
    if "".join(matches) != remaining:
        print(f"{option}: Invalid option", file=sstm.stderr)
        parser.print_help()
        sstm.exit(-1)

    coordinate_idc = [row_idx, col_idx]
    for match in matches:
        coordinate_idc.append(csio.ColLabelToIdx(match[1:]))

    return coordinate_idc, row_transform


def _ArgumentParser() -> argument_parser_t:
    """"""
    output = argument_parser_t(
        # prog=path_t(main_package.__file__).stem,
        description=DESCRIPTION,
        formatter_class=RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    output.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    output.add_argument(
        "--gt",
        type=str,
        dest="ground_truth_path",
        metavar="Ground_truth",
        help="Ground-truth CSV file of centers or labeled image or labeled Numpy array, or ground-truth folder.",
    )
    output.add_argument(
        "--r|x<capital letter(s)>c|y<capital letter(s)>",
        action="store_true",
        help="If Ground_truth is a CSV file, then an option of the form --rAcB (or --xAyB) can be passed additionally "
        "to indicate that columns A and B contain the centers' "
        "rows and cols, respectively (or x's and y's in x/y mode). "
        "Columns must be specified as uppercase letters (e.g., A for the first column, or AB for the 28th one), "
        'as is usual in spreadsheet applications. For ground-truths of dimension "n" higher than 2, '
        'the symbol "+" must be used for the remaining "n-2" dimensions. For example, --rAcB+C+D in dimension 4.',
    )
    output.add_argument(
        "--relabel-gt",
        type=str,
        choices=("seq", "full"),
        default=None,
        dest="relabel_gt",
        help="If present, this option instructs to relabel the ground-truth with sequential labels (seq),"
        "or to fully relabel the non-zero regions of the ground-truth with maximum connectivity (full).",
    )
    output.add_argument(
        "--dn",
        type=str,
        dest="detection_path",
        metavar="Detection",
        help="Detection labeled image or labeled Numpy array, or detection folder.",
    )
    output.add_argument(
        "--relabel-dn",
        type=str,
        choices=("seq", "full"),
        default=None,
        dest="relabel_dn",
        help='Equivalent of the "--relabel-gt" option for the detection.',
    )
    output.add_argument(
        "--shifts",
        type=int,
        nargs="+",
        action="extend",
        default=None,
        dest="dn_shifts",
        metavar="Dn_shift",
        help="Vertical (row), horizontal (col), and higher dimension shifts to apply to detection. "
        "Default: all zeroes.",
    )
    output.add_argument(
        "-e",
        "--exclude-border",
        action="store_true",
        dest="should_exclude_border",
        help="If present, this option instructs to discard objects touching image border, "
        "both in ground-truth and detection.",
    )
    output.add_argument(
        "-t",
        "--tol",
        "--tolerance",
        type=float,
        default=0.0,
        dest="tolerance",
        help="Max ground-truth-to-detection distance to count as a hit "
        "(meant to be used when ground-truth is a CSV file of centers). Default: zero.",
    )
    output.add_argument(
        "-f",
        "--format",
        type=str,
        choices=("csv", "nev"),
        default="nev",
        dest="output_format",
        help='nev: one "Name = Value"-row per measure; '
        'csv: one CSV-row per ground-truth/detection pairs. Default: "nev".',
    )
    output.add_argument(
        "-o",
        type=otpt.OutputStream,  # Do not use the same approach with gt and dn since they can be folders
        default=sstm.stdout,
        dest="output_accessor",
        metavar="Output_file",
        help='CSV file to store the computed measures or "-" for console output. Default: console output.',
    )
    output.add_argument(
        "-s",
        "--show-image",
        action="store_true",
        dest="should_show_image",
        help="If present, this option instructs to show an image "
        "superimposing ground-truth onto detection. It is actually done only for 2-dimensional images.",
    )
    output.add_argument(
        "--explain-measures",
        action="store_true",
        dest="should_explain_measures",
        help="Prints accuracy measure definitions and exits.",
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
