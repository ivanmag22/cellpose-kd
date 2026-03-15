"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

import dominate.tags as html
from session.constants import PROJECT_NAME as NAME

NAME_MEANING = "Detection and Segmentation Accuracy Measures"
SHORT_DESCRIPTION = "Computation of some Standard Performance Measures of Object Detection and Segmentation."

ABOUT = html.div(
    html.h4("Purpose"),
    html.p(SHORT_DESCRIPTION),
    #
    html.h4("Privacy"),
    html.p(
        f"""
            Uploaded inputs are stored on the server for the sole processing by {NAME}.
            All data, uploaded, intermediate and final, can be deleted manually through a dedicated button
            that appears on processing completion.
            If not deleted manually, all data are automatically deleted based on their timestamp
            by a process running several times a day.
        """
    ),
    #
    html.h4("Links"),
    html.ul(
        html.li(
            html.a(
                "Documentation",
                href="https://src.koda.cnrs.fr/eric.debreuve/daccuracy/-/wikis/home",
            )
        ),
        html.li(
            html.a(
                "Source code (Gitlab repository)",
                href="https://src.koda.cnrs.fr/eric.debreuve/daccuracy/",
            )
        ),
        html.li(
            html.a(
                "Python Package Index (PyPI) page",
                href="https://pypi.org/project/daccuracy/",
            )
        ),
    ),
)

MAX_IMAGE_SIZE = 256  # MB


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
