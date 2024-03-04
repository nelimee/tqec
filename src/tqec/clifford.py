from __future__ import annotations

import typing as ty
from copy import deepcopy
from dataclasses import dataclass

import cirq
import numpy
import stim
import stimcirq

from tqec.exceptions import TQECException


def _split_circuit_by_gates(
    circuit: cirq.Circuit,
    gate_type: type,
) -> ty.Iterator[cirq.Circuit | cirq.Moment]:
    """Split a Circuit instance.

    This function splits a Circuit according to a provided ``gate_type``.

    Its unique pre-condition is quite simple: any Moment in the provided ``circuit``
    that contains an operation matching the provided ``gate_type`` should only
    contain operations that match that same ``gate_type``.

    Args:
        circuit: the Circuit instance to split
        gate_type: type inheriting from cirq.Gate that will be used to match
            operations that should be considered as splitting the circuit. Any
            operation ``op`` that checks ``isinstance(op.gate, gate_type)`` is
            splitting the ``circuit``.

    Returns:
        a list of:
        - ``cirq.Circuit`` instances, representing the split circuit portions not
            containing any operation matching the provided ``gate_type``.
        - ``cirq.Moment`` instances grouping the operations matching ``gate_type``.

    Raises:
        TQECError: if any cirq.Moment in the provided ``circuit`` contains at least
            one operation that matches with the provided ``gate_type`` AND one that
            do not match.
    """
    first_moment_of_section: int = 0
    for i, moment in enumerate(circuit.moments):
        is_optype_gate_type = [
            isinstance(op.gate, gate_type) for op in moment.operations
        ]
        if any(is_optype_gate_type):
            if not all(is_optype_gate_type):
                raise TQECException(
                    "Found a Moment with both 1) one operation with a "
                    f"gate type of {gate_type} and 2) one operation with "
                    f"a gate type that is not {gate_type}."
                )
            # If i == 0, this means that the first Moment instance contains gates
            # that we split on, so the first cirq.Circuit instance is empty. As it
            # adds no value to include an empty Circuit, filter it out here.
            if i > 0:
                yield circuit[first_moment_of_section:i]
            yield circuit.moments[i]
            first_moment_of_section = i + 1
    # There might be some Moment instances left, in which case add
    # them to the circuit.
    if first_moment_of_section < len(circuit.moments):
        yield circuit[first_moment_of_section:]


@dataclass
class Stabiliser:
    pauli: stim.PauliString

    def to_pauli_string(self) -> str:
        raw_str = self.pauli.__str__()
        return raw_str[1:].replace("_", "I")

    @staticmethod
    def from_pauli_string(pauli_string: str) -> Stabiliser:
        return Stabiliser(stim.PauliString(pauli_string))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_pauli_string()})"


class CollapsingOperation:
    def __init__(
        self,
        stabiliser: Stabiliser,
        is_creation: bool,
    ) -> None:
        """Create a collapsing operation representing either measuremements or resets.

        This class represents what is called a collapsing operation in "Stim: a fast
        stabilizer circuit simulator", Craig Gidney, https://arxiv.org/abs/2103.02202,
        https://doi.org/10.22331/q-2021-07-06-497.
        It is basically either a measurement or a reset, depending on the value of
        ``is_creation``.

        Args:
            stabiliser: operator that is stabilised by the operation.
            is_creation: True if the collapsing operation produces a state stabilised
                by the stabiliser represented by the provided ``moment`` (i.e., a reset).
                Else (i.e., in the case of a measurement), False.
        """
        self._stabiliser = stabiliser
        self._is_creation = is_creation

    @staticmethod
    def from_moment(
        moment: cirq.Moment,
        qubit_map: ty.Mapping[cirq.Qid, int],
        is_creation: bool,
    ) -> CollapsingOperation:
        """Create a collapsing operation representing either a measuremement or a reset.

        This class represents what is called a collapsing operation in "Stim: a fast
        stabilizer circuit simulator", Craig Gidney, https://arxiv.org/abs/2103.02202,
        https://doi.org/10.22331/q-2021-07-06-497.
        It is basically either a measurement or a reset, depending on the value of
        ``is_creation``.

        Args:
            moment: a cirq.Moment instance containing either 1) only measurements or
                2) only resets. Due to cirq limitations, these can only be in the Z
                basis for the moment.
            qubit_map: a map to translate cirq.Qid instances (i.e., cirq qubits) to
                integer indices (i.e., Stim qubits).
            is_creation: True if the collapsing operation produces a state stabilised
                by the stabiliser represented by the provided ``moment`` (i.e., a reset).
                Else (i.e., in the case of a measurement), False.
        """
        stabiliser_pauli = stim.PauliString(max(qubit_map.values()) + 1)
        for collapsing_operation in moment:
            if len(collapsing_operation.qubits) != 1:
                raise TQECException(
                    f"Found an invalid collapsing operation: {collapsing_operation}. "
                    "Collapsing operations should be applied on exactly one qubit!"
                )
            (qubit,) = collapsing_operation.qubits
            if qubit not in qubit_map:
                raise TQECException(
                    f"Collapsing operation on invalid qubit found. "
                    f"Collapsing operation: {collapsing_operation}. Invalid qubit: {qubit}."
                )
            # cirq can only represent Z-basis measurements and resets, so we know that
            # the stabilized state is the +1 eigenstate of Z.
            stabiliser_pauli[qubit_map[qubit]] = "Z"
        return CollapsingOperation(Stabiliser(stabiliser_pauli), is_creation)

    def inverse(self, *, unsigned: bool = False) -> CollapsingOperation:
        """Inverse the collapsing operation, fliping the value of ``is_creation``."""
        cpy = deepcopy(self)
        cpy._is_creation = not cpy._is_creation
        return cpy

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.stabiliser}', {self.is_creation})"

    @property
    def is_creation(self) -> bool:
        return self._is_creation

    @property
    def stabiliser(self) -> Stabiliser:
        return self._stabiliser

    def apply_on(self, stabilised_state: Stabiliser) -> Stabiliser:
        """Apply the collapsing operation on a stabilised state."""
        result_pauli_string: str = ""
        for pauli, collapsing_pauli in zip(
            stabilised_state.to_pauli_string(), self._stabiliser.to_pauli_string()
        ):
            # On a non-collapsing operation on that qubit, keep the pauli
            # given as input
            if collapsing_pauli == "I":
                result_pauli_string += pauli
                continue

            if self.is_creation:
                if pauli != "I":
                    raise TQECException(
                        "Cannot apply a reset-like collapsing operation on "
                        "non-identity Pauli."
                    )
                # On a reset, the resulting Pauli takes the reset basis.
                result_pauli_string += collapsing_pauli
            else:
                if pauli != collapsing_pauli:
                    raise TQECException(
                        "Cannot apply a measurement-like collapsing operation "
                        "on a non-commuting Pauli."
                    )
                # On a measurement, the resulting Pauli becomes the identity
                result_pauli_string += "I"

        return Stabiliser.from_pauli_string(result_pauli_string)


class TableauWithCollapsingOperations:
    def __init__(
        self,
        operations: ty.Sequence[stim.Tableau | CollapsingOperation],
        qubit_map: ty.Mapping[int, cirq.Qid],
    ) -> None:
        self._operations = operations
        self._qubit_map = qubit_map

    @staticmethod
    def from_circuit(circuit: cirq.Circuit) -> TableauWithCollapsingOperations:
        qubit_map = {q: i for i, q in enumerate(circuit.all_qubits())}
        operations: list[stim.Tableau | CollapsingOperation] = []
        for element_with_maybe_resets in _split_circuit_by_gates(
            circuit, cirq.MeasurementGate
        ):
            if isinstance(element_with_maybe_resets, cirq.Moment):
                operations.append(
                    CollapsingOperation.from_moment(
                        element_with_maybe_resets, qubit_map, is_creation=False
                    )
                )
                continue
            for element in _split_circuit_by_gates(
                element_with_maybe_resets, cirq.ResetChannel
            ):
                if isinstance(element, cirq.Circuit):
                    operations.append(
                        stim.Tableau.from_circuit(
                            stimcirq.cirq_circuit_to_stim_circuit(
                                element, qubit_to_index_dict=qubit_map
                            )
                        )
                    )
                elif isinstance(element, cirq.Moment):
                    operations.append(
                        CollapsingOperation.from_moment(
                            element, qubit_map, is_creation=True
                        )
                    )
        return TableauWithCollapsingOperations(
            operations, {i: q for q, i in qubit_map.items()}
        )

    def inverse(self) -> TableauWithCollapsingOperations:
        return TableauWithCollapsingOperations(
            # TODO: check if we need the sign or not...
            [op.inverse() for op in reversed(self._operations)],
            self._qubit_map,
        )

    @property
    def qubit_number(self) -> int:
        return max(self._qubit_map.keys()) + 1

    def output_stabiliser(
        self, input_stabiliser: Stabiliser | None = None
    ) -> Stabiliser:
        if input_stabiliser is None:
            input_stabiliser = Stabiliser(stim.PauliString(self.qubit_number))

        for op in self._operations:
            if isinstance(op, CollapsingOperation):
                input_stabiliser = op.apply_on(input_stabiliser)
            elif isinstance(op, stim.Tableau):
                input_stabiliser.pauli = input_stabiliser.pauli.after(
                    op, targets=self._qubit_map.keys()
                )

        return input_stabiliser
