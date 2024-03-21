from __future__ import annotations

import typing as ty
from dataclasses import dataclass

import networkx as nx
import numpy
import numpy.typing as npty
import stim
from pysat.formula import IDPool
from pysat.solvers import CryptoMinisat
from tqec.circuit.clifford.stabiliser import Stabiliser
from tqec.exceptions import TQECException


@dataclass(frozen=True)
class CollapsingOperationLocation:
    time_coordinate: int
    space_coordinate: int
    basis: ty.Literal["X", "Y", "Z"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.time_coordinate}, {self.space_coordinate}, {self.basis})"


class CollapsingOperationsInteractionMap:
    def __init__(
        self,
        measurements: ty.Sequence[CollapsingOperationLocation] | None = None,
        resets: ty.Sequence[CollapsingOperationLocation] | None = None,
    ) -> None:
        self._map = nx.DiGraph()
        self._stabilisers: dict[CollapsingOperationLocation, dict[int, Stabiliser]] = {}

        if measurements is not None:
            self.add_operations(measurements, are_measurement=True)
        if resets is not None:
            self.add_operations(resets, are_measurement=False)

    def add_operation(
        self, op: CollapsingOperationLocation, is_measurement: bool
    ) -> None:
        self._map.add_node(op, is_measurement=is_measurement)

    def add_operations(
        self,
        operations: ty.Sequence[CollapsingOperationLocation],
        are_measurement: bool,
    ) -> None:
        for op in operations:
            self.add_operation(op, are_measurement)

    def _add_stabiliser(
        self,
        measurement: CollapsingOperationLocation,
        reset_time_coordinate: int,
        stabiliser: Stabiliser,
    ) -> None:
        measurement_stabilisers = self._stabilisers.setdefault(measurement, dict())
        measurement_stabilisers[reset_time_coordinate] = stabiliser

    def add_connected_resets(
        self,
        measurement: CollapsingOperationLocation,
        resets: ty.Sequence[CollapsingOperationLocation],
        stabiliser: Stabiliser | None = None,
    ) -> None:
        if not resets:
            return

        for collapsing_op in [measurement, *resets]:
            if collapsing_op not in self._map:
                raise TQECException(
                    f"Trying to add a connection involving {collapsing_op} "
                    "that has never been added to the interaction map."
                )
        reset_time_coordinate = {r.time_coordinate for r in resets}
        if len(reset_time_coordinate) != 1:
            raise TQECException(
                "All the resets provided to one call of add_connected_resets should "
                "have the same time coordinate. Found the following coordinates: "
                f"{reset_time_coordinate}."
            )
        reset_time_coordinate = reset_time_coordinate.pop()
        if reset_time_coordinate >= measurement.time_coordinate:
            raise TQECException(
                f"A reset happening at time {reset_time_coordinate} cannot "
                f"influence a measurement happening at time {measurement.time_coordinate}."
            )

        for reset in resets:
            self._map.add_edge(reset, measurement)

        if stabiliser is not None:
            self._add_stabiliser(measurement, reset_time_coordinate, stabiliser)

    @property
    def interaction_graph(self) -> nx.Graph:
        return self._map

    def stabilisers_at_timeslice_for_measurement(
        self, timeslice: int, measurement: CollapsingOperationLocation
    ) -> Stabiliser:
        if measurement not in self._stabilisers:
            raise TQECException(
                f"Could not find measurement {measurement} in the interaction map."
            )
        if timeslice not in self._stabilisers[measurement]:
            raise TQECException(
                f"Could not find timeslice {timeslice} for measurement {measurement} "
                "in the interaction map."
            )
        return self._stabilisers[measurement][timeslice]

    @property
    def measurements(self) -> ty.Iterator[CollapsingOperationLocation]:
        yield from (
            collapsing_op
            for collapsing_op, is_measurement in self._map.nodes(data="is_measurement")
            if is_measurement
        )

    @property
    def resets(self) -> ty.Iterator[CollapsingOperationLocation]:
        yield from (
            collapsing_op
            for collapsing_op, is_measurement in self._map.nodes(data="is_measurement")
            if not is_measurement
        )

    @property
    def reset_timeslices(self) -> set[int]:
        return {reset.time_coordinate for reset in self.resets}

    @property
    def measurement_timeslices(self) -> set[int]:
        return {measure.time_coordinate for measure in self.measurements}

    def get_measurements_within(
        self, start: int, stop: int
    ) -> ty.Iterator[CollapsingOperationLocation]:
        yield from filter(
            lambda op: start <= op.time_coordinate < stop, self.measurements
        )

    def get_resets_within(
        self, start: int, stop: int
    ) -> ty.Iterator[CollapsingOperationLocation]:
        yield from filter(lambda op: start <= op.time_coordinate < stop, self.resets)

    def get_measurements_potentially_impacting(
        self, measurement: CollapsingOperationLocation
    ) -> ty.Iterable[CollapsingOperationLocation]:
        resets = filter(
            lambda r: r.time_coordinate < measurement.time_coordinate,
            self._map.predecessors(measurement),
        )
        measurements_impacted_by_resets = [set(self._map.successors(r)) for r in resets]
        return set().union(*measurements_impacted_by_resets)


def find_detectors(imap: CollapsingOperationsInteractionMap) -> None:
    for measurement_timeslice in imap.measurement_timeslices:
        measurements_without_detectors = set(
            imap.get_measurements_within(
                measurement_timeslice, measurement_timeslice + 1
            )
        )
        for reset_timeslice in filter(
            lambda t: t < measurement_timeslice, imap.reset_timeslices
        ):
            measurements_to_consider = list(
                imap.get_measurements_within(reset_timeslice, measurement_timeslice + 1)
            )
            resets_to_consider = list(
                imap.get_resets_within(reset_timeslice, reset_timeslice + 1)
            )
            measurement2index = {
                m: i + 1 for i, m in enumerate(measurements_to_consider)
            }
            index2measurement = {v: k for k, v in measurement2index.items()}
            # Build the boolean matrix expected to construct the SAT problem.
            measurement_pauli_effect_on_qubit = numpy.array(
                [
                    imap.stabilisers_at_timeslice_for_measurement(
                        reset_timeslice, m
                    ).to_numpy_per_qubit()
                    for m in measurements_to_consider
                ],
                dtype=bool,
            )
            # Build the vector of expected pauli effects.
            _, qubit_number, _ = measurement_pauli_effect_on_qubit.shape
            expected_reset_pauli_string = stim.PauliString(qubit_number)
            for reset in resets_to_consider:
                expected_reset_pauli_string[reset.space_coordinate] = reset.basis
            expected_pauli_effect = Stabiliser(
                expected_reset_pauli_string
            ).to_numpy_per_qubit()

            with CryptoMinisat() as solver:
                encode_generic_sat_problem_in_solver(
                    solver, measurement_pauli_effect_on_qubit, expected_pauli_effect
                )
                measurements_with_detectors = set()
                for measurement in measurements_without_detectors:
                    impacting_measurements = set(
                        imap.get_measurements_potentially_impacting(measurement)
                    )
                    assumptions = []
                    # The currently tested measurement should be part of the detector.
                    assumptions.append(measurement2index[measurement])
                    # Any measurement that cannot have an impact on the resets touched by
                    # the currently tested measurement cannot be part of the detector.
                    assumptions.extend(
                        [
                            -measurement2index[m]
                            for m in measurements_to_consider
                            if m not in impacting_measurements
                        ]
                    )
                    # Try to solve:
                    if solver.solve(assumptions=assumptions):
                        solution = solver.get_model()
                        measurements_in_detector = [i for i in solution if i > 0]
                        print(
                            f"Found detector for {measurement}: {[index2measurement[m] for m in measurements_in_detector]}"
                        )
                    else:
                        print(f"No detector found for measurement {measurement}")


def encode_generic_sat_problem_in_solver(
    solver: CryptoMinisat,
    measurement_pauli_effect_on_qubit: npty.NDArray[numpy.bool_],
    expected_pauli_effect: npty.NDArray[numpy.bool_],
):
    """Build the SAT problem that should be solved to find detectors.

    This function builds the SAT problem, that will then have to be solved to
    find detectors (or the absence of detectors). The built SAT problem is
    generic in the sense that it does NOT represent the problem to solve
    to find ONE detector. Users are expected to fix some of the boolean
    variables to actually find the detector including a particular measurement.

    Boolean variables representing measurements are indexed according to the
    order in ``measurement_pauli_effect_on_qubit``. Their indexing starts at 1
    (instead of the default 0-indexing) because pyset solvers expect 1-based
    indexing for boolean variables.

    This function has 2 main integer parameters that are of interest:
    - ``m`` that is the number of measurements considered.
    - ``q`` that is the number of qubits considered.

    The resulting SAT problem is a conjunction of XOR clauses that may
    contain up to:
    - ``m`` atoms if there is no Y-basis reset,
    - ``2m`` atoms if there is at least one Y-basis reset.

    Args:
        solver: the solver on which to encode the SAT problem.
        measurement_pauli_effect_on_qubit: a numpy array of shape ``(m, q, 2)``
            and containing boolean values with:
            - the first dimension representing the measurement,
            - the second dimension representing the qubit,
            - the third dimension representing the effect in the X (index 0) or
              Z (index 1) basis.

            ``measurement_pauli_effect_on_qubit[i, j, k]`` is ``true`` if the i-th
            measurement has the effect X (resp. Z) if ``k`` is 0 (resp 1) on the j-th
            qubit.
        expected_pauli_effect: a numpy array of shape ``(q, 2)`` that represent the
            pauli effect that should be present on qubit ``q`` to be valid
            (i.e. commute). An identity effect ([0, 0] on the 2nd dimension of the
            array) is always considered possible, so the valid effect on qubit ``q``
            will be ``expected_pauli_effect[q]`` or ``[0, 0]``.
    """
    X, Z = 0, 1
    measurement_number, qubit_number, _ = measurement_pauli_effect_on_qubit.shape
    for q in range(qubit_number):
        expected_effect = tuple(expected_pauli_effect[q])
        if expected_effect == (0, 0):
            # Both the X and Z effect on that qubit should be OFF (i.e., identity).
            for P in (X, Z):
                lits = [
                    m + 1
                    for m in range(measurement_number)
                    if measurement_pauli_effect_on_qubit[m, q, P]
                ]
                solver.add_xor_clause(lits, value=False)

        elif expected_effect == (1, 0):
            # The X effect should be ON, the Z effect should be OFF.
            # Because the identity (0, 0) is also valid, X can be either ON or OFF,
            # so we do not need to restrict X effect.
            lits = [
                m + 1
                for m in range(measurement_number)
                if measurement_pauli_effect_on_qubit[m, q, Z]
            ]
            solver.add_xor_clause(lits, value=False)

        elif expected_effect == (0, 1):
            # The X effect should be OFF, the Z effect should be ON.
            # Because the identity (0, 0) is also valid, Z can be either ON or OFF,
            # so we do not need to restrict Z effect.
            lits = [
                m + 1
                for m in range(measurement_number)
                if measurement_pauli_effect_on_qubit[m, q, X]
            ]
            solver.add_xor_clause(lits, value=False)

        elif expected_effect == (1, 1):
            # This is an expected Y effect. This means that X and Z should be both
            # ON or both OFF. Rephrasing using XOR, this is equivalent to
            # X_effect XOR Z_effect == 0.
            # Note that measurements appearing twice here can be removed, as:
            # - the XOR operation is commutative (so we can re-organise the measurements
            #   to be sorted by index).
            # - b XOR b = 0.
            # - b XOR 0 = b.
            # Also, a given measurement cannot appear more than twice, so we are fine
            # just removing duplicated entries.
            X_clause = {
                m + 1
                for m in range(measurement_number)
                if measurement_pauli_effect_on_qubit[m, q, X]
            }
            Z_clause = {
                m + 1
                for m in range(measurement_number)
                if measurement_pauli_effect_on_qubit[m, q, Z]
            }
            lits = [m for m in X_clause.symmetric_difference(Z_clause)]
            solver.add_xor_clause(lits, value=False)
