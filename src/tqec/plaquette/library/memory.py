from __future__ import annotations

from tqec.plaquette.enums import PlaquetteOrientation
from tqec.plaquette.library.pauli import pauli_memory_plaquette
from tqec.plaquette.library.utils.schedule import cnot_pauli_schedule
from tqec.plaquette.plaquette import Plaquette
from tqec.plaquette.qubit import RoundedPlaquetteQubits, SquarePlaquetteQubits


def xx_memory_plaquette(
    orientation: PlaquetteOrientation, schedule: list[int] | None = None
) -> Plaquette:
    """R - H - CX - CX - H - M"""
    if schedule is None:
        schedule = [1, 2] + cnot_pauli_schedule("xx", orientation) + [7, 8]

    return pauli_memory_plaquette(RoundedPlaquetteQubits(orientation), "XX", schedule)


def xxxx_memory_plaquette(schedule: list[int] | None = None) -> Plaquette:
    """R - H - CX - CX - CX - CX - H - M"""
    if schedule is None:
        schedule = [1, 2, 3, 4, 5, 6, 7, 8]

    return pauli_memory_plaquette(SquarePlaquetteQubits(), "XXXX", schedule)


def zz_memory_plaquette(
    orientation: PlaquetteOrientation, schedule: list[int] | None = None
) -> Plaquette:
    """R - CX - CX - M"""
    if schedule is None:
        schedule = [1] + cnot_pauli_schedule("zz", orientation) + [8]

    return pauli_memory_plaquette(RoundedPlaquetteQubits(orientation), "ZZ", schedule)


def zzzz_memory_plaquette(schedule: list[int] | None = None) -> Plaquette:
    """R - CX - CX - CX - CX - M"""
    if schedule is None:
        schedule = [1, 3, 4, 5, 6, 8]

    return pauli_memory_plaquette(
        SquarePlaquetteQubits().permute_data_qubits([0, 2, 1, 3]), "ZZZZ", schedule
    )
