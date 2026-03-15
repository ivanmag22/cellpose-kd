"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import datetime as dttm
import secrets as scrt
from configparser import ConfigParser as ini_config_t
from pathlib import Path as path_t

import dominate.tags as html
import flask as flsk
from generic import URLOfPath
from html_.constants import ABOUT, MAX_IMAGE_SIZE, NAME, NAME_MEANING
from html_.session import SessionAsHTML, SessionOutputsAsHTML
from session.form import input_form_t
from session.processing import ProcessSession
from session.session import INI_SECTION, session_t
from flask_bootstrap import Bootstrap as BootstrapFlask
from flask_session import Session as flask_session_t

HTML_FOLDER = "html_"
HTML_TEMPLATE = "main.html"
BASE_FOLDER = "static"

RUNTIME_FOLDER = path_t(BASE_FOLDER) / "runtime"
INPUT_FOLDER = RUNTIME_FOLDER / "input"  # Used as upload folder
OUTPUT_FOLDER = RUNTIME_FOLDER / "output"
SESSION_FOLDER = RUNTIME_FOLDER / "session"


GLOBAL_APP = flsk.Flask(__name__, template_folder=HTML_FOLDER)


def HomePage(
    session_id: str, /, *, session: session_t = None, form: input_form_t = None
) -> str:
    """"""
    return flsk.render_template(
        HTML_TEMPLATE,
        name=NAME,
        name_meaning=NAME_MEANING,
        about=ABOUT,
        session=SessionAsHTML(session, session_id),
        form=form,
        max_file_size=MAX_IMAGE_SIZE,
        outputs=SessionOutputsAsHTML(session),
        data_management=_DataManagement(session, session_id),
    )


def _DataManagement(session: session_t, session_id: str, /) -> html.html_tag | None:
    """"""
    if session is None:
        return None

    if (path := session.outputs_path) is None:
        output_url = None
    else:
        output_url = URLOfPath(path)

    output = html.div()
    with output:
        if output_url is not None:
            html.a(
                html.button(
                    "Download Result",
                    type="button",
                    _class="btn btn-primary",
                    style="margin-top: 8pt; margin-bottom: 12pt",
                ),
                href=output_url,
                download="",
            )
            html.span(style="margin-right:48pt")
        html.a(
            html.button(
                html.b("Clear All Data"),
                type="button",
                _class="btn btn-primary",
                style="margin-top: 8pt; margin-bottom: 12pt",
            ),
            href=flsk.url_for(".DeleteSession", session_id=session_id),
        )

    return output


@GLOBAL_APP.route("/")
def LaunchNewSession() -> flsk.Response:
    """"""
    session_id = scrt.token_urlsafe()
    flsk.session[session_id] = session_t()

    return flsk.redirect(f"/{session_id}")


@GLOBAL_APP.route("/<session_id>", methods=("GET", "POST"))
def UpdateSessionPage(*, session_id: str = None) -> str:
    """"""
    session = flsk.session[session_id]
    form = input_form_t()  # Do not pass flask.request.form

    if flsk.request.method == "GET":
        form.Update(session.AsDictionary())

        if session.is_complete:
            return HomePage(session_id, session=session, form=form)
    elif form.validate_on_submit():
        submission = form.Submission(INPUT_FOLDER)
        session.UpdateInputs(submission, form.file_fields)

        if session.is_complete:
            outputs = ProcessSession(session, session_id, OUTPUT_FOLDER)
            session.UpdateOutputs(*outputs)

            return HomePage(session_id, session=session, form=form)

    return HomePage(session_id, session=session, form=form)


@GLOBAL_APP.route("/load/<session_id>", methods=("POST",))
def LoadSession(*, session_id: str = None) -> flsk.Response:
    """"""
    session = flsk.session[session_id]

    session_file = tuple(flsk.request.files.values())[0]
    new_as_str = session_file.read().decode("utf-8")
    new_as_ini = ini_config_t()
    new_as_ini.read_string(new_as_str)

    # Form file fields are not file_t's then, so the loaded session cannot be used as is
    for field, value in new_as_ini.items(INI_SECTION):
        session[field] = value

    return flsk.redirect(f"/{session_id}")


@GLOBAL_APP.route("/save/<session_id>")
def SaveSession(*, session_id: str = None) -> flsk.Response:
    """"""
    session = flsk.session[session_id]

    name, path = _SessionNameAndPath(session_id)
    with open(path, "w") as accessor:
        session.AsINI().write(accessor)

    return flsk.send_file(
        path, mimetype="text/plain", as_attachment=True, download_name=name
    )


@GLOBAL_APP.route("/delete/<session_id>")
def DeleteSession(*, session_id: str = None) -> flsk.Response:
    """"""
    session = flsk.session[session_id]

    session.DeleteInputFiles()
    session.DeleteOutputsFile()
    _DeleteSessionFile(session_id)

    flsk.session.pop(session_id, None)

    return flsk.redirect("/")


def _DeleteSessionFile(session_id: str, /) -> None:
    """"""
    _, path = _SessionNameAndPath(session_id)
    if path.is_file():
        path.unlink()


def _SessionNameAndPath(session_id: str, /) -> tuple[str, path_t]:
    """"""
    name = f"session-{session_id}.ini"
    path = OUTPUT_FOLDER / name

    return name, path


if __name__ == "__main__":
    #
    GLOBAL_APP.config.from_mapping(
        PREFERRED_URL_SCHEME="https",
        SESSION_TYPE="filesystem",
        PERMANENT_SESSION_LIFETIME=dttm.timedelta(hours=1),
        SESSION_FILE_DIR=SESSION_FOLDER,
        SECRET_KEY=scrt.token_bytes(),
        MAX_CONTENT_LENGTH=MAX_IMAGE_SIZE * 1024 * 1024,
    )
    flask_session_t(GLOBAL_APP)

    BootstrapFlask(GLOBAL_APP)
    GLOBAL_APP.run(debug=True)


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
