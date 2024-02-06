from tqec.templates.scale import Dimension, LinearFunction


def test_dimension_init():
    dim = Dimension(5, LinearFunction())
    assert dim.value == 5


def test_dimension_default_scaling():
    dim = Dimension(2, scaling_function=LinearFunction(2))
    dim.scale_to(3)
    assert dim.value == 2 * 3


def test_dimension_scaling():
    scaling_func = LinearFunction(3, 4)

    dim = Dimension(2, scaling_func)
    dim.scale_to(4)
    assert dim.value == scaling_func(4)
    assert dim.scale_to(19).value == scaling_func(19)