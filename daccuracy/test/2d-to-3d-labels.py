"""
Copyright CNRS/Inria/UniCA
Contributor(s): Eric Debreuve (eric.debreuve@cnrs.fr) since 2019
SEE COPYRIGHT NOTICE BELOW
"""

# Run from project folder with: python -m daccuracy.test.test-2d-2-3d-label

from pathlib import Path as path_t
from tempfile import TemporaryDirectory

import matplotlib.pyplot as pypl
import numpy as nmpy
import skimage.draw as data
import skimage.io as skio
import skimage.morphology as mrph
from daccuracy.cli.two_d_labeled_slices_to_3d import LabeledImage

n_shapes = 20
n_images = 20

with TemporaryDirectory() as folder:
    image, _ = data.random_shapes(
        (300, 400),
        n_shapes,
        min_shapes=n_shapes,
        min_size=3,
        max_size=5,
        channel_axis=None,
        shape="ellipse",
        intensity_range=(1, n_shapes),
    )
    image = mrph.label(image != 255)
    labels = tuple(nmpy.unique(image))[1:]

    folder_as_path = path_t(folder)
    n_decimal_places = str(n_images).__len__()
    for i_idx in range(n_images):
        new_image = nmpy.zeros_like(image)
        new_labels = nmpy.random.permutation(labels)
        assert tuple(sorted(new_labels)) == labels
        for current_label, new_label in zip(labels, new_labels):
            new_image[image == current_label] = new_label
        assert tuple(nmpy.unique(new_image)) == (0,) + labels

        i_idx_as_str = str(i_idx).zfill(n_decimal_places)
        skio.imsave(
            folder_as_path / f"image-{i_idx_as_str}.png", new_image.astype(nmpy.uint16)
        )

    labeled = LabeledImage(folder_as_path, None)

for s_idx in range(labeled.shape[2]):
    pypl.figure()
    pypl.imshow(labeled[..., s_idx])
pypl.show()


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
