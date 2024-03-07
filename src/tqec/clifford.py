from __future__ import annotations

import typing as ty
from copy import deepcopy
from dataclasses import dataclass

import cirq
import stim
import stimcirq

from tqec.circuit.operations.operation import STIM_TAG
from tqec.exceptions import TQECException


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
        It is basically either measurements or resets, depending on the value of
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
        operations: ty.Sequence[
            stim.Tableau | CollapsingOperation | TableauWithCollapsingOperations
        ],
        qubit_map: ty.Mapping[int, cirq.Qid],
        repetitions: int = 1,
    ) -> None:
        self._operations = operations
        self._qubit_map = qubit_map
        self._repetitions = repetitions

    @staticmethod
    def from_circuit(
        circuit: cirq.Circuit, repetitions: int = 1
    ) -> TableauWithCollapsingOperations:
        qubit_map = {q: i for i, q in enumerate(circuit.all_qubits())}
        operations: list[
            list[cirq.Moment] | CollapsingOperation | TableauWithCollapsingOperations
        ] = []
        for moment in circuit.moments:
            is_measurement = [
                isinstance(op.gate, cirq.MeasurementGate) for op in moment.operations
            ]
            is_reset = [
                isinstance(op.gate, cirq.ResetChannel) for op in moment.operations
            ]
            is_circuit_operation = [
                isinstance(op.untagged, cirq.CircuitOperation)
                for op in moment.operations
            ]
            # Moments composed of only measurements.
            if any(is_measurement):
                if not all(is_measurement):
                    raise TQECException("Measurements should be in their own Moment.")
                operations.append(
                    CollapsingOperation.from_moment(
                        moment, qubit_map, is_creation=False
                    )
                )
            # Moments composed of only resets.
            elif any(is_reset):
                if not all(is_reset):
                    raise TQECException("Resets should be in their own Moment.")
                operations.append(
                    CollapsingOperation.from_moment(moment, qubit_map, is_creation=True)
                )
            # Moments composed of ONE CircuitOperation instance.
            elif any(is_circuit_operation):
                if len(moment.operations) > 1:
                    raise TQECException(
                        "CircuitOperation instance should be in its own Moment."
                    )
                circuit_op: cirq.CircuitOperation = moment.operations[0].untagged
                operations.append(
                    TableauWithCollapsingOperations.from_circuit(
                        circuit_op.circuit.unfreeze(), int(circuit_op.repetitions)
                    )
                )
            # Regular operations
            else:
                if not isinstance(operations[-1], list):
                    operations.append(list())
                operations[-1].append(moment)
        tableau_operations: list[
            CollapsingOperation | TableauWithCollapsingOperations | stim.Tableau
        ] = []
        for op in operations:
            if not isinstance(op, list):
                tableau_operations.append(op)
                continue
            tableau = stim.Tableau.from_circuit(
                stimcirq.cirq_circuit_to_stim_circuit(cirq.Circuit(op))
            )
            if len(tableau) != 0:
                tableau_operations.append(tableau)

        return TableauWithCollapsingOperations(
            tableau_operations,
            {i: q for q, i in qubit_map.items()},
            repetitions=repetitions,
        )

    def inverse(self) -> TableauWithCollapsingOperations:
        return TableauWithCollapsingOperations(
            # TODO: check if we need the sign or not...
            [op.inverse() for op in reversed(self._operations)],
            self._qubit_map,
            self._repetitions,
        )

    def apply_on(self, stabilised_state: Stabiliser) -> Stabiliser:
        # TODO: might be optimised by finding a fixed point if there is a repetition.
        for _ in range(self._repetitions):
            for op in self._operations:
                if isinstance(op, CollapsingOperation):
                    stabilised_state = op.apply_on(stabilised_state)
                elif isinstance(op, stim.Tableau):
                    stabilised_state.pauli = stabilised_state.pauli.after(
                        op, targets=self._qubit_map.keys()
                    )
                elif isinstance(op, TableauWithCollapsingOperations):
                    stabilised_state = op.apply_on(stabilised_state)
        return stabilised_state

    @property
    def qubit_number(self) -> int:
        return max(self._qubit_map.keys()) + 1

    def output_stabiliser(
        self, input_stabiliser: Stabiliser | None = None
    ) -> Stabiliser:
        if input_stabiliser is None:
            input_stabiliser = Stabiliser(stim.PauliString(self.qubit_number))

        return self.apply_on(input_stabiliser)
