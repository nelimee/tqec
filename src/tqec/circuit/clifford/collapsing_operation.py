from __future__ import annotations

import typing as ty
from copy import deepcopy

import cirq
import stim
from tqec.circuit.clifford.stabiliser import Stabiliser
from tqec.exceptions import TQECException


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
                        f"on a non-commuting Pauli. Measurement in the {collapsing_pauli}"
                        f"-basis applied on a stabiliser {pauli}."
                    )
                # On a measurement, the resulting Pauli becomes the identity
                result_pauli_string += "I"

        return Stabiliser.from_pauli_string(result_pauli_string)

    @property
    def effects(self) -> dict[int, str]:
        return {
            i: p for i, p in enumerate(self.stabiliser.to_pauli_string()) if p != "I"
        }
