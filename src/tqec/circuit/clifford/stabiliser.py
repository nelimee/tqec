from __future__ import annotations

from dataclasses import dataclass

import stim


@dataclass
class Stabiliser:
    pauli: stim.PauliString

    def to_pauli_string(self) -> str:
        raw_str = self.pauli.__str__()
        return raw_str[1:].replace("_", "I")

    @staticmethod
    def from_pauli_string(pauli_string: str) -> Stabiliser:
        return Stabiliser(stim.PauliString(pauli_string))

    @staticmethod
    def from_1q_stabiliser(
        stabiliser: str, index: int, qubit_number: int
    ) -> Stabiliser:
        assert index < qubit_number
        assert len(stabiliser) == 1
        return Stabiliser.from_pauli_string(
            "_" * (index) + stabiliser + "_" * (qubit_number - index - 1)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_pauli_string()})"

    def acts_non_trivially_on_qubit(self, qubit: int) -> bool:
        return self.pauli[qubit] != 0
