from __future__ import annotations
from dataclasses import dataclass

import numbers
import typing as ty

import cirq

from tqec.circuit.circuit import generate_circuit
from tqec.circuit.operations import make_shift_coords
from tqec.plaquette.plaquette import Plaquette
from tqec.templates.base import Template


def make_repeated_layer(circuit: cirq.Circuit, repetitions: int) -> cirq.Circuit:
    """Build and return a repeated circuit

    This method inserts ``ShiftCoords(0, 0, 1)`` operations at the end of each
    repetitions.

    Args:
        circuit: quantum circuit that will be repeated.
        repetitions: number of repetitions to use.

    Returns:
        a ``cirq.Circuit`` instance that behaves as the provided ``circuit`` repeated
        ``repetitions`` times.
    """
    circuit_to_repeat = circuit + cirq.Circuit(cirq.Moment(make_shift_coords(0, 0, 1)))
    repeated_circuit_operation = cirq.CircuitOperation(
        circuit_to_repeat.freeze()
    ).repeat(repetitions)
    return cirq.Circuit([repeated_circuit_operation])


class QECLayer:
    def __init__(
        self,
        template: Template,
        plaquettes: ty.Sequence[Plaquette] | ty.Mapping[int, Plaquette],
        repetitions: int = 1,
    ) -> None:
        assert repetitions >= 1 and isinstance(repetitions, numbers.Integral)

        self._template = template
        self._plaquettes = plaquettes
        self._repetitions = repetitions

    def generate_circuit(self, k: int) -> cirq.Circuit:
        raw_circuit = generate_circuit(self._template.scale_to(k), self._plaquettes)
        shift_op_in_circuit = cirq.Circuit(cirq.Moment(make_shift_coords(0, 0, 1)))
        return (
            raw_circuit
            if self._repetitions == 1
            else make_repeated_layer(raw_circuit, self._repetitions)
        ) + shift_op_in_circuit


class QECLayers:
    def __init__(self, layers: list[QECLayer]) -> None:
        self._layers = layers

    