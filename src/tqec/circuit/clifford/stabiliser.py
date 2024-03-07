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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_pauli_string()})"
