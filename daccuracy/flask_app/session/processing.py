"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

from io import StringIO as io_string_t
from pathlib import Path as path_t

from daccuracy.cli.arguments import ProcessedArguments
from daccuracy.main import ComputeAndOutputMeasures
from .constants import APP_NAME as CMD_NAME
from .form import FORMAT_CHOICES, RELABELING_CHOICES
from .session import session_t

_RELABELING_TRANSLATION = {"None": None, "Sequential": "seq", "Full": "full"}
_FORMAT_TRANSLATION = {"Name = Value": "nev", "CSV": "csv"}

if sorted(_RELABELING_TRANSLATION.keys()) != sorted(RELABELING_CHOICES):
    raise ValueError(
        f"{tuple(_RELABELING_TRANSLATION.keys())}: Translation keys do not match form choices {RELABELING_CHOICES}"
    )
if sorted(_FORMAT_TRANSLATION.keys()) != sorted(FORMAT_CHOICES):
    raise ValueError(
        f"{tuple(_FORMAT_TRANSLATION.keys())}: Translation keys do not match form choices {FORMAT_CHOICES}"
    )


def ProcessSession(
    session: session_t, session_id: str, folder: path_t, /
) -> tuple[tuple[str, str], path_t | None]:
    """"""
    arguments = [
        "--gt",
        str(session["reference"].path),
        "--dn",
        str(session["detection"].path),
    ]
    relabeling = _RELABELING_TRANSLATION[session["relabel_rf"]]
    if relabeling is not None:
        arguments.extend(("--relabel-gt", relabeling))
    relabeling = _RELABELING_TRANSLATION[session["relabel_dn"]]
    if relabeling is not None:
        arguments.extend(("--relabel-dn", relabeling))
    if session["shifts"] != "":
        arguments.extend(("--shifts", session["shifts"]))
    if session["exclude_border"]:
        arguments.append("--exclude-border")
    if session["tolerance"] != 0:
        arguments.extend(("--tolerance", str(session["tolerance"])))
    output_format = _FORMAT_TRANSLATION[session["output_format"]]
    arguments.extend(("--format", output_format))
    # if session["show_image"]:
    #     arguments.append("--save-image")

    options = ProcessedArguments(arguments)
    fake_file = io_string_t()
    options["output_accessor"] = fake_file

    ComputeAndOutputMeasures(options)

    measures = fake_file.getvalue()
    fake_file.close()

    arguments[1] = session["reference"].name
    arguments[3] = session["detection"].name
    cmd_line = f"{CMD_NAME} " + " ".join(arguments)

    if output_format == "csv":
        extension = "csv"
    else:
        extension = "txt"
    path = folder / f"measures-{session_id}.{extension}"
    with open(path, "w") as accessor:
        accessor.write(measures)

    return (cmd_line, measures), path


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
