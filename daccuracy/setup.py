"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import re as rgex
from importlib import util
from pathlib import Path as path_t

from setuptools import find_namespace_packages, setup

HERE = path_t(__file__).parent.resolve()


def DescriptionFromAsciidoc() -> dict[str, str]:
    """"""
    output = {}

    where = HERE / "documentation" / "wiki" / "description.asciidoc"
    pattern = rgex.compile(r":([A-Z_]+): +(.+)\n?", flags=rgex.ASCII)

    with open(where) as accessor:
        for line in accessor.readlines():
            if (match := pattern.fullmatch(line)) is not None:
                name = match.group(1)
                value = match.group(2)
                output[name] = value

    return output


DESCRIPTION = DescriptionFromAsciidoc()
ENTRY_POINTS = {
    "console_scripts": [
            f"daccuracy={DESCRIPTION['IMPORT_NAME']}.cli.main:Main",
            f"two_d_labeled_slices_to_3d={DESCRIPTION['IMPORT_NAME']}.cli.two_d_labeled_slices_to_3d:Main",
        ],
}


long_description = (HERE / "README.rst").read_text(encoding="utf-8")

repository_url = (
    f"https://"
    f"{DESCRIPTION['REPOSITORY_SITE']}/"
    f"{DESCRIPTION['REPOSITORY_USER']}/"
    f"{DESCRIPTION['REPOSITORY_NAME']}/"
)
documentation_url = f"{repository_url}{DESCRIPTION['DOCUMENTATION_SITE']}"


def Version() -> str:
    """"""
    where = HERE / "package" / DESCRIPTION["IMPORT_NAME"] / "version.py"
    spec = util.spec_from_file_location(where.stem, where)
    module = spec.loader.load_module(spec.name)

    output = module.__version__
    if isinstance(output, str) and rgex.fullmatch(r"20[0-9]{2}\.[1-9][0-9]*", output):
        return output

    raise ValueError(f"{output}: Invalid version.")


def Requirements() -> tuple[str, ...]:
    """"""
    if not (HERE / "requirements.txt").is_file():
        return ()

    with open(HERE / "requirements.txt") as accessor:
        output = accessor.readlines()

    return tuple(output)


if __name__ == "__main__":
    #
    # fmt: off
    setup(
        author=DESCRIPTION["AUTHOR"],
        author_email=DESCRIPTION["EMAIL"],
        #
        name=DESCRIPTION["PYPI_NAME"],
        description=DESCRIPTION["SHORT_DESCRIPTION"],
        long_description=long_description,
        long_description_content_type="text/x-rst",
        license=DESCRIPTION["LICENSE_SHORT"],
        version=Version(),
        #
        classifiers=[
            f"Topic :: {DESCRIPTION['PYPI_TOPIC']}",
            f"Intended Audience :: {DESCRIPTION['PYPI_AUDIENCE']}",
            f"License :: OSI Approved :: {DESCRIPTION['LICENCE_LONG']} ({DESCRIPTION['LICENSE_SHORT']})",
            f"Programming Language :: Python :: {DESCRIPTION['PY_VERSION_MAJOR']}",
            f"Development Status :: {DESCRIPTION['PYPI_STATUS']}",
        ],
        keywords=DESCRIPTION["KEYWORDS"],
        #
        url=repository_url,
        project_urls={
            "Documentation": documentation_url,
            "Source": repository_url,
        },
        #
        packages=find_namespace_packages(where="package"),
        package_dir={"": "package"},
        entry_points=ENTRY_POINTS,
        python_requires=f">={DESCRIPTION['PY_VERSION_MIN']}",
        install_requires=Requirements(),
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
