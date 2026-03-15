"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import typing as h
from configparser import ConfigParser as ini_config_t
from pathlib import Path as path_t

from .constants import APP_NAME as INI_SECTION
from .form import file_t


class session_t(dict):
    form_file_fields: tuple[str, ...] = None
    outputs: h.Any = None
    outputs_path: path_t | None = None

    def UpdateInputs(
        self, inputs: dict[str, h.Any], file_fields: h.Sequence[str], /
    ) -> None:
        """"""
        if not set(file_fields).issubset(inputs.keys()):
            raise ValueError(
                f"{file_fields}: Form file fields not a subset of form fields {tuple(inputs.keys())}"
            )

        for field, value in inputs.items():
            if value is not None:
                self[field] = value

        self.form_file_fields = tuple(file_fields)

    def InputFilesContents(
        self, LoadInput: h.Callable[[path_t], h.Any], /
    ) -> dict[str, h.Any]:
        """
        Example:
            inputs = session.InputFilesContents(skimage.io.imread)
        """
        return {_inp: LoadInput(self[_inp][1]) for _inp in self.form_file_fields}

    def DeleteInputFiles(self) -> None:
        """"""
        if self.form_file_fields is None:
            return

        for field in self.form_file_fields:
            path = self[field].path
            if path.is_file():
                path.unlink()

    def UpdateOutputs(self, outputs: h.Any, outputs_path: path_t | None, /) -> None:
        """"""
        self.outputs = outputs
        self.outputs_path = outputs_path

    def DeleteOutputsFile(self) -> None:
        """"""
        if ((path := self.outputs_path) is not None) and path.is_file():
            path.unlink()
            self.UpdateOutputs(None, None)

    def AsDictionary(self) -> dict[str, h.Any]:
        """"""
        return dict(self)

    def AsINI(self) -> ini_config_t:
        """
        Does include form file fields but filenames only since paths of input files cannot be stored in a session as
        they are only known by the client.
        """
        output = ini_config_t()

        output[INI_SECTION] = {
            _fld: _val.name if isinstance(_val, file_t) else _val
            for _fld, _val in self.items()
        }

        return output

    @property
    def is_empty(self) -> bool:
        """"""
        return self.__len__() == 0

    @property
    def is_complete(self) -> bool:
        """"""
        # Do not use self[_key] below since reference and/or detection files are missing if the form has been submitted
        # without these files (they are not required fields since the session can supply them) and the session has not
        # received these files yet, e.g. on the first run if not selecting these files.
        return (
            (not self.is_empty)
            and all(_val is not None for _val in self.values())
            and (self.form_file_fields is not None)
            and all(
                isinstance(self.get(_key), file_t) for _key in self.form_file_fields
            )
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
