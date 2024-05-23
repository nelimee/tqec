import typing as ty

import pydantic

from .atomic import (
    AlternatingCornerSquareTemplate,
    AlternatingRectangleTemplate,
    AlternatingSquareTemplate,
    RawRectangleTemplate,
)
from .base import Template, TemplateWithIndices
from .composed import ComposedTemplate
from .constructions import (
    QubitRectangleTemplate,
    QubitSquareTemplate,
    ScalableCorner,
)
from .display import (
    display_template,
    display_templates_svg,
)
from .scale import Dimension, FixedDimension, LinearFunction

"""Type representing the different instantiable templates.

This type should be updated by each Template subclass that
should be an instantiable pydantic model.
See https://github.com/pydantic/pydantic/discussions/4208 for
more information on why this is done like that.

## Important note

Due to the type system not being able to really provide an 
"empty type union" (that can be populated by the subclasses), 
the initial type is using typing.NoReturn 
(https://docs.python.org/3/library/typing.html#typing.NoReturn)
that should behave like the empty type. This means that the below 
type is initialised to a type that is equivalent to `typing.Union[]`
(i.e., no types), even if that is not a well formated type with
Python's typing system.
"""
AnyTemplate = ty.Union[
    ComposedTemplate,
    AlternatingRectangleTemplate,
    AlternatingSquareTemplate,
    RawRectangleTemplate,
]
DiscriminatedAnyTemplate = ty.Annotated[
    AnyTemplate, pydantic.Field(discriminator="tag")
]
