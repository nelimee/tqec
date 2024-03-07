from __future__ import annotations

import typing as ty

import cirq
import stim
import stimcirq
from cirq.circuits.circuit_operation import INT_TYPE
from tqec.circuit.clifford.collapsing_operation import CollapsingOperation
from tqec.circuit.clifford.stabiliser import Stabiliser
from tqec.exceptions import TQECException


def _is_measurement_moment(moment: cirq.Moment) -> bool:
    is_measurement = [
        isinstance(op.gate, cirq.MeasurementGate) for op in moment.operations
    ]
    if any(is_measurement) and not all(is_measurement):
        raise TQECException("Measurements should be in their own Moment.")
    return any(is_measurement)


def _is_reset_moment(moment: cirq.Moment) -> bool:
    is_reset = [isinstance(op.gate, cirq.ResetChannel) for op in moment.operations]
    if any(is_reset) and not all(is_reset):
        raise TQECException("Resets should be in their own Moment.")
    return any(is_reset)


def _is_circuit_operation_moment(moment: cirq.Moment) -> bool:
    is_circuit_operation = [
        isinstance(op.untagged, cirq.CircuitOperation) for op in moment.operations
    ]
    if any(is_circuit_operation) and len(moment.operations) > 1:
        raise TQECException("CircuitOperation instance should be in its own Moment.")
    return any(is_circuit_operation)


def _consecutive_moments_to_tableaus(
    operations: list[
        list[cirq.Moment] | CollapsingOperation | TableauWithCollapsingOperations
    ],
) -> list[stim.Tableau | CollapsingOperation | TableauWithCollapsingOperations]:
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
    return tableau_operations


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
            # Moments composed of only measurements.
            if _is_measurement_moment(moment):
                operations.append(
                    CollapsingOperation.from_moment(
                        moment, qubit_map, is_creation=False
                    )
                )
            # Moments composed of only resets.
            elif _is_reset_moment(moment):
                operations.append(
                    CollapsingOperation.from_moment(moment, qubit_map, is_creation=True)
                )
            # Moments composed of ONE CircuitOperation instance.
            elif _is_circuit_operation_moment(moment):
                circuit_op: cirq.CircuitOperation = moment.operations[0]  # type: ignore
                # Avoid sympy.Expr for the moment. TODO: handle it properly in the future.
                assert isinstance(circuit_op.repetitions, INT_TYPE)
                subtableau = TableauWithCollapsingOperations.from_circuit(
                    circuit_op.circuit.unfreeze(), int(circuit_op.repetitions)
                )
                operations.append(subtableau)
            # Regular operations
            else:
                if not isinstance(operations[-1], list):
                    operations.append(list())
                operations[-1].append(moment)  # type: ignore

        tableau_operations = _consecutive_moments_to_tableaus(operations)

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

    def propagate_forward_each_creation(
        self, include_stabilisers_at_destruction: bool = False
    ):
        pass
