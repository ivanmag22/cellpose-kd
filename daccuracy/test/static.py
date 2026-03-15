"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

# Run from project folder with: python -m daccuracy.test.test-static

import subprocess as prss
from pathlib import Path as path_t

MODULE = "daccuracy.cli.daccuracy"
BASE_FOLDER = path_t(__file__).parent
if "site-packages" in BASE_FOLDER.parts:
    print(
        "Script only for internal use; "
        'Can still be run by copying to a user folder and changing the 2 folders "GT_FOLDER" and "DN_FOLDER".'
    )
    import sys as sstm

    sstm.exit(0)
GT_FOLDER = BASE_FOLDER / "resource" / "ground-truth"
DN_FOLDER = BASE_FOLDER / "resource" / "detection"


csv_vs_correct = (
    f"python -m {MODULE} "
    f"--gt {GT_FOLDER / 'ground-truth-1.csv'} "
    f"--dn {DN_FOLDER / 'detection-1.png'} "
    f"-s --format csv"
)
csv_vs_wrong = (
    f"python -m {MODULE} "
    f"--gt {GT_FOLDER / 'ground-truth-1.csv'} "
    f"--dn {DN_FOLDER / 'detection-2.png'} "
    f"--rAcB "
    f"-s --format nev "
)
csv_vs_wrong_w_tol = (
    f"python -m {MODULE} "
    f"--gt {GT_FOLDER / 'ground-truth-1.csv'} "
    f"--dn {DN_FOLDER / 'detection-2.png'} "
    f"-t 5 "
    f"-s --format nev "
)
png_vs_wrong = (
    f"python -m {MODULE} "
    f"--gt {GT_FOLDER / 'ground-truth-1.png'} "
    f"--dn {DN_FOLDER / 'detection-2.png'} "
    f"-s --format csv "
)

print(prss.run(csv_vs_correct.split(), capture_output=True).stdout.decode("utf-8"))
print("\n")
print(prss.run(csv_vs_wrong.split(), capture_output=True).stdout.decode("utf-8"))
print("\n")
print(prss.run(csv_vs_wrong_w_tol.split(), capture_output=True).stdout.decode("utf-8"))
print("\n")
print(prss.run(png_vs_wrong.split(), capture_output=True).stdout.decode("utf-8"))


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
