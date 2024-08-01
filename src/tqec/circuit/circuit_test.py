from collections import defaultdict

import pytest

from tqec.circuit.circuit import generate_circuit
from tqec.circuit.schedule import ScheduledCircuit
from tqec.exceptions import TQECException
from tqec.plaquette.enums import PlaquetteOrientation
from tqec.plaquette.library import zz_memory_plaquette
from tqec.plaquette.plaquette import Plaquette
from tqec.templates.atomic.rectangle import RawRectangleTemplate
from tqec.templates.base import Template


@pytest.fixture
def plaquette() -> Plaquette:
    return zz_memory_plaquette(PlaquetteOrientation.LEFT, [1, 5, 6, 8])


@pytest.fixture
def one_by_one_template() -> Template:
    return RawRectangleTemplate([[0]])


def untagged(circuit: ScheduledCircuit) -> ScheduledCircuit:
    return ScheduledCircuit(
        circuit.raw_circuit.map_operations(lambda op: op.untagged), circuit.schedule
    )


def test_generate_circuit_list(
    plaquette: Plaquette, one_by_one_template: Template
) -> None:
    circuit = generate_circuit(one_by_one_template, [plaquette])
    assert circuit == untagged(plaquette.circuit)


def test_generate_circuit_dict(
    plaquette: Plaquette, one_by_one_template: Template
) -> None:
    circuit = generate_circuit(one_by_one_template, {1: plaquette})
    assert circuit == untagged(plaquette.circuit)


def test_generate_circuit_defaultdict(
    plaquette: Plaquette, one_by_one_template: Template
) -> None:
    circuit = generate_circuit(one_by_one_template, defaultdict(lambda: plaquette))
    assert circuit == untagged(plaquette.circuit)


def test_generate_circuit_dict_0_indexed(
    plaquette: Plaquette, one_by_one_template: Template
) -> None:
    with pytest.raises(TQECException):
        generate_circuit(one_by_one_template, {0: plaquette})


def test_generate_circuit_wrong_number_of_plaquettes(
    plaquette: Plaquette, one_by_one_template: Template
) -> None:
    with pytest.raises(TQECException):
        generate_circuit(one_by_one_template, [plaquette, plaquette])
    with pytest.raises(TQECException):
        generate_circuit(one_by_one_template, [])
