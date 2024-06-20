import itertools
import typing

import pytest
import stim
from tqec.circuit.detectors.pauli import PauliString
from tqec.circuit.detectors.utils import (
    collapse_pauli_strings_at_moment,
    iter_stim_circuit_by_moments,
    split_combined_measurement_reset,
    split_combined_measurement_reset_in_moment,
)
from tqec.exceptions import TQECException


def test_iter_by_moment_empty():
    circuit = stim.Circuit()
    assert len(list(iter_stim_circuit_by_moments(circuit))) == 0


def test_iter_by_moment_single_tick():
    circuit = stim.Circuit("TICK")
    assert len(list(iter_stim_circuit_by_moments(circuit))) == 1
    first_moment = next(iter_stim_circuit_by_moments(circuit))
    assert isinstance(first_moment, stim.Circuit)
    assert len(first_moment) == 1
    assert first_moment == circuit


def test_iter_by_moment_single_qec_round():
    circuit = stim.Circuit("""
        R 0 1 2 3 4
        TICK
        CX 0 1 2 3
        TICK
        CX 2 1 4 3
        TICK
        MR 1 3
        DETECTOR(1, 0) rec[-2]
        DETECTOR(3, 0) rec[-1]
        M 0 2 4
        DETECTOR(1, 1) rec[-2] rec[-3] rec[-5]
        DETECTOR(3, 1) rec[-1] rec[-2] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-1]""")
    assert len(list(iter_stim_circuit_by_moments(circuit))) == 4
    moment_iterator = iter_stim_circuit_by_moments(circuit)
    moment = next(moment_iterator)
    assert isinstance(moment, stim.Circuit)
    assert len(moment) == 2
    assert moment == stim.Circuit("R 0 1 2 3 4\nTICK")
    moment = next(moment_iterator)
    assert isinstance(moment, stim.Circuit)
    assert len(moment) == 2
    assert moment == stim.Circuit("CX 0 1 2 3\nTICK")
    moment = next(moment_iterator)
    assert isinstance(moment, stim.Circuit)
    assert len(moment) == 2
    assert moment == stim.Circuit("CX 2 1 4 3\nTICK")
    moment = next(moment_iterator)
    assert isinstance(moment, stim.Circuit)
    assert len(moment) == 7
    assert moment == stim.Circuit("""MR 1 3
        DETECTOR(1, 0) rec[-2]
        DETECTOR(3, 0) rec[-1]
        M 0 2 4
        DETECTOR(1, 1) rec[-2] rec[-3] rec[-5]
        DETECTOR(3, 1) rec[-1] rec[-2] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-1]""")


def test_iter_by_moment_repeat_block():
    circuit = stim.Circuit("""
        REPEAT 9 {
            TICK
            CX 0 1 2 3
            TICK
            CX 2 1 4 3
            TICK
            MR 1 3
            SHIFT_COORDS(0, 1)
            DETECTOR(1, 0) rec[-2] rec[-4]
            DETECTOR(3, 0) rec[-1] rec[-3]
        }""")
    assert len(list(iter_stim_circuit_by_moments(circuit))) == 1
    moment = next(iter_stim_circuit_by_moments(circuit))
    assert isinstance(moment, stim.CircuitRepeatBlock)
    assert len(moment.body_copy()) == 9
    circuit = stim.Circuit("""
        R 0 1 2 3 4
        TICK 
        CX 0 1 2 3
        TICK
        CX 2 1 4 3
        TICK
        MR 1 3
        REPEAT 9 {
            TICK
            CX 0 1 2 3
            TICK
            CX 2 1 4 3
            TICK
            MR 1 3
            SHIFT_COORDS(0, 1)
            DETECTOR(1, 0) rec[-2] rec[-4]
            DETECTOR(3, 0) rec[-1] rec[-3]
        }""")
    # REPEAT block should NOT be with the last MR before it, even without TICK.
    assert len(list(iter_stim_circuit_by_moments(circuit))) == 5


def test_split_combined_measurement_reset_in_moment_simple():
    for has_tick, basis in itertools.product([True, False], ["", "X", "Y", "Z"]):
        tick = "\nTICK" if has_tick else ""
        moment = stim.Circuit(f"MR{basis} 1 2 3 4" + tick)
        measurements, resets = split_combined_measurement_reset_in_moment(moment)
        assert measurements == stim.Circuit(f"M{basis} 1 2 3 4\nTICK")
        assert resets == stim.Circuit(f"R{basis} 1 2 3 4" + tick)


def test_split_combined_measurement_reset_in_moment_annotated():
    moment = stim.Circuit("""
        MR 1 3
        SHIFT_COORDS(0, 1)
        DETECTOR(1, 0) rec[-2] rec[-4]
        DETECTOR(3, 0) rec[-1] rec[-3]""")
    measurements, resets = split_combined_measurement_reset_in_moment(moment)
    assert measurements == stim.Circuit("""
        M 1 3
        SHIFT_COORDS(0, 1)
        DETECTOR(1, 0) rec[-2] rec[-4]
        DETECTOR(3, 0) rec[-1] rec[-3]
        TICK""")
    assert resets == stim.Circuit("R 1 3")


def test_split_combined_measurement_reset_in_moment_raising():
    moment = stim.Circuit("REPEAT 9 {\n    MR 1 2 3 4\n}")
    with pytest.raises(
        TQECException, match=r".*stim.CircuitRepeatBlock instances are not.*"
    ):
        split_combined_measurement_reset_in_moment(moment)

    moment = stim.Circuit("""
        X_ERROR(0.1) 1 3
        MR 1 3
        X_ERROR(0.1) 1 3 4""")  # 4 has not been measured.
    with pytest.raises(
        TQECException,
        match=r"^Found an instruction applied on partially measured qubits.*",
    ):
        split_combined_measurement_reset_in_moment(moment)

    moment = stim.Circuit("""
        R 0 2 4
        MR 1 3
        M 0 2 4""")  # 4 has not been measured.
    with pytest.raises(
        TQECException,
        match=r"^Breaking pre-condition: found a non-combined measurement or reset gate\.$",
    ):
        split_combined_measurement_reset_in_moment(moment)


def test_split_combined_measurement_reset_in_moment_with_noisy_gates():
    moment = stim.Circuit("""
        X_ERROR(0.1) 1 3
        MR 1 3
        X_ERROR(0.1) 1 3""")
    measurements, resets = split_combined_measurement_reset_in_moment(moment)
    assert measurements == stim.Circuit("""
        X_ERROR(0.1) 1 3
        M 1 3
        TICK""")
    assert resets == stim.Circuit("R 1 3\nX_ERROR(0.1) 1 3")


def test_split_combined_measurement_reset():
    circuit = stim.Circuit("""
        R 0 1 2 3 4
        TICK
        CX 0 1 2 3
        TICK
        CX 2 1 4 3
        TICK
        MR 1 3
        REPEAT 9 {
            TICK
            CX 0 1 2 3
            TICK
            CX 2 1 4 3
            TICK
            MR 1 3
        }
        M 0 2 4""")
    assert split_combined_measurement_reset(circuit) == stim.Circuit("""
        R 0 1 2 3 4
        TICK
        CX 0 1 2 3
        TICK
        CX 2 1 4 3
        TICK
        M 1 3
        TICK
        R 1 3
        REPEAT 9 {
            TICK
            CX 0 1 2 3
            TICK
            CX 2 1 4 3
            TICK
            M 1 3
            TICK
            R 1 3
        }
        M 0 2 4""")


def test_collapse_pauli_strings_at_moment_all_basis():
    qubits = list(range(5))
    qubits_str = " ".join(map(str, qubits))
    for basis in ["", "X", "Y", "Z"]:
        circuit = stim.Circuit(f"R{basis} {qubits_str}")
        collapsing_operations = collapse_pauli_strings_at_moment(circuit)
        pauli: typing.Literal["X", "Y", "Z"] = "Z" if basis == "" else basis  # type:ignore
        assert set(collapsing_operations) == {PauliString({q: pauli}) for q in qubits}


def test_collapse_pauli_strings_at_moment_virtual_instruction():
    circuit = stim.Circuit("R 0 1 2\nX_ERROR(0.01) 0 1 2")
    collapsing_operations = collapse_pauli_strings_at_moment(circuit)
    assert set(collapsing_operations) == {PauliString({q: "Z"}) for q in [0, 1, 2]}
