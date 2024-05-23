import typing as ty

import numpy
from pydantic.dataclasses import dataclass

from tqec.position import Displacement, Shape2D
from tqec.templates.base import Template
from tqec.templates.scale import Dimension


@dataclass
class ScalableOffset:
    x: Dimension
    y: Dimension

    def scale_to(self, k: int) -> None:
        self.x.scale_to(k)
        self.y.scale_to(k)


class ShiftedTemplate(Template):
    shifted_template: Template
    offset: ScalableOffset

    def __init__(
        self,
        template: Template,
        offset: ScalableOffset,
        default_x_increment: int = 2,
        default_y_increment: int = 2,
    ) -> None:
        super().__init__(default_x_increment, default_y_increment)
        self.shifted_template = template
        self.offset = offset

    def scale_to(self, k: int) -> "ShiftedTemplate":
        self.shifted_template.scale_to(k)
        self.offset.scale_to(k)
        return self

    @property
    def shape(self) -> Shape2D:
        tshape = self.shifted_template.shape
        return Shape2D(self.offset.x.value + tshape.x, self.offset.y.value + tshape.y)

    @property
    def expected_plaquettes_number(self) -> int:
        return self.shifted_template.expected_plaquettes_number

    def instantiate(self, plaquette_indices: ty.Sequence[int]) -> numpy.ndarray:
        # Do not explicitely check here, the check is forwarded to the
        # shifted Template instance.
        arr = numpy.zeros(self.shape.to_numpy_shape(), dtype=int)
        tshape = self.shifted_template.shape
        xoffset, yoffset = self.offset.x.value, self.offset.y.value
        tarr = self.shifted_template.instantiate(plaquette_indices)
        arr[yoffset : yoffset + tshape.y, xoffset : xoffset + tshape.x] = tarr
        return arr
