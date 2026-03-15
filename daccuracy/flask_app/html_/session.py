"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import dominate.tags as html
import flask as flsk
from session.form import file_t, input_form_t
from session.session import session_t


def SessionAsHTML(session: session_t | None, session_id: str, /) -> html.html_tag:
    """"""
    output = html.div(_class="container")

    empty_form = input_form_t()
    file_fields = empty_form.file_fields
    field_names_to_labels = empty_form.field_names_to_labels

    if (is_none := (session is None)) or session.is_empty:
        if is_none:
            session = {}
        save = None
    else:
        save = html.a(
            html.button(
                "Save Session",
                _class="btn btn-primary",
                style="margin-right:24pt; margin-top: 12pt; margin-bottom: 24pt",
            ),
            href=flsk.url_for(".SaveSession", session_id=session_id),
        )
    load = _SessionLoadingForm(session_id)

    row = None
    for idx, (name, label) in enumerate(field_names_to_labels.items()):
        value = session.get(name, "")
        if isinstance(value, file_t):
            value = value.name
        elif isinstance(value, str) and (value.__len__() > 0) and (name in file_fields):
            # The value has been assigned by a session loading, which does not produce valid form file fields
            value = html.span(
                f"{value} (must be re-uploaded)",
                style="color:Crimson; font-weight:bold",
            )

        if idx % 2 == 0:
            if row is not None:
                output.add(row)
            row = html.div(_class="row")
        row.add(html.div(f"{label}: ", value, _class="col"))

    if row is not None:
        output.add(row)

    table = html.table()
    with table:
        with html.tr():
            if save is not None:
                html.td(save)
            html.td(load)
    output.add(table)

    return output


def SessionOutputsAsHTML(session: session_t, /) -> html.html_tag | None:
    """
    Needs to be kept in sync with processing.py since it assigns the outputs
    """
    if (outputs := session.outputs) is None:
        return None

    cmd_line, measures = outputs  # This is where syncing matters
    output = html.div()
    with output:
        html.p(f"Equivalent command line: {cmd_line}")
        if session["output_format"] == "CSV":
            with html.table(border="1px"):
                lines = measures.splitlines()
                with html.tr():
                    for element in lines[0].split(";"):
                        html.th(element.strip(), style="vertical-align: top")
                for line in lines[1:]:
                    with html.tr():
                        for element in line.split(";"):
                            html.td(element.strip(), style="vertical-align: top")
        else:
            html.pre(measures)

    return output


def _SessionLoadingForm(session_id: str, /) -> html.form:
    """"""
    output = html.form(
        role="form",
        method="post",
        enctype="multipart/form-data",
        action=flsk.url_for(".LoadSession", session_id=session_id),
        style="margin-bottom: 12pt",
    )

    with output:
        html.label("Load Session")
        html.input_(
            type="file",
            name="session",
            required=True,
            _class="btn btn-primary",
            style="margin-top: 12pt; margin-bottom: 12pt",
        )
        html.input_(
            type="submit",
            name="submit",
            value="Validate",
            _class="btn btn-primary",
            style="margin-top: 12pt; margin-bottom: 12pt",
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
