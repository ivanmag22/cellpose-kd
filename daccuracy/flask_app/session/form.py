"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import datetime as dttm
import re as regx
import typing as h
from pathlib import Path as path_t

import wtforms as wtfm
from .constants import PROJECT_NAME as NAME
from flask_wtf import FlaskForm as flask_form_t
from werkzeug.utils import secure_filename as SecureFilenameVersion

validators_t = wtfm.validators


class file_t(h.NamedTuple):
    name: str
    path: path_t


_FALSE_VALUES = (False, "False", "false")
RELABELING_CHOICES = ("None", "Sequential", "Full")
FORMAT_CHOICES = ("Name = Value", "CSV")
SHIFTS_FORMAT = regx.compile("^(\d+( +\d+)*)?$")


class input_form_t(flask_form_t):
    """"""

    reference = wtfm.FileField(label="Reference")
    detection = wtfm.FileField(label="Detection")

    relabel_rf = wtfm.SelectField(
        label="Reference relabeling", choices=RELABELING_CHOICES
    )
    relabel_dn = wtfm.SelectField(
        label="Detection relabeling", choices=RELABELING_CHOICES
    )

    tolerance = wtfm.DecimalField(
        label="Tolerance",
        default=0.0,
        validators=(
            validators_t.NumberRange(
                min=0.0, message="Must greater than or equal to zero"
            ),
        ),
    )
    shifts = wtfm.StringField(
        label="Detection shifts",
        render_kw={"placeholder": "Space-separated list of integers"},
        validators=(
            validators_t.Regexp(
                SHIFTS_FORMAT,
                message="Can be empty or must be a space-separated list of integers",
            ),
        ),
    )
    exclude_border = wtfm.BooleanField(
        label="Exclude border", default=True, false_values=_FALSE_VALUES
    )

    output_format = wtfm.SelectField(
        label="Output format", default=FORMAT_CHOICES[0], choices=FORMAT_CHOICES
    )
    show_image = wtfm.BooleanField(
        label="Show image", default=False, false_values=_FALSE_VALUES
    )

    submit = wtfm.SubmitField(label=f"Execute {NAME}")

    @property
    def field_names_to_labels(self) -> dict[str, str]:
        """"""
        output = {}

        for name in self.__dict__:
            attribute = getattr(self, name)
            if _ElementIsInputField(attribute):  # Not all elements are fields
                # Fields might not have a label (at least it does not cost much to check)
                if hasattr(attribute, "label"):
                    output[name] = attribute.label.text
                else:
                    output[name] = name

        return output

    @property
    def file_fields(self) -> tuple[str, ...]:
        """"""
        output = []

        for name in self.__dict__:
            if isinstance(getattr(self, name), wtfm.FileField):
                output.append(name)

        return tuple(output)

    def Update(self, session: dict[str, h.Any], /) -> None:
        """"""
        for field, value in session.items():
            if not isinstance(value, file_t):
                getattr(self, field).process_formdata((value,))

    def Submission(self, upload_folder: str | path_t, /) -> dict[str, h.Any]:
        """"""
        output = {}

        if isinstance(upload_folder, str):
            upload_folder = path_t(upload_folder)
        time_stamp = (
            dttm.datetime.now()
            .isoformat(timespec="microseconds")
            .translate(str.maketrans(":.", "--"))
        )

        for name in self.__dict__:
            attribute = getattr(self, name)

            if _ElementIsInputField(attribute):
                data = attribute.data
                if isinstance(attribute, wtfm.FileField):
                    if (filename := data.filename) == "":
                        output[name] = None
                    else:
                        secure = SecureFilenameVersion(filename)
                        path = upload_folder / f"{time_stamp}-{secure}"
                        data.save(path)
                        output[name] = file_t(name=filename, path=path)
                else:
                    output[name] = data

        return output


def _ElementIsInputField(element: h.Any, /) -> bool:
    """"""
    return isinstance(element, wtfm.Field) and not isinstance(
        element, (wtfm.HiddenField, wtfm.SubmitField)
    )


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
