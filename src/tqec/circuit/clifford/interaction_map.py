from __future__ import annotations

import typing as ty
from dataclasses import dataclass

import networkx as nx
from tqec.circuit.clifford.stabiliser import Stabiliser
from tqec.exceptions import TQECException


@dataclass(frozen=True)
class CollapsingOperationLocation:
    time_coordinate: int
    space_coordinate: int


class CollapsingOperationsInteractionMap:
    def __init__(
        self,
        measurements: ty.Sequence[CollapsingOperationLocation] | None = None,
        resets: ty.Sequence[CollapsingOperationLocation] | None = None,
    ) -> None:
        self._map = nx.Graph()
        self._stabilisers: dict[CollapsingOperationLocation, dict[int, Stabiliser]] = {}

        if measurements is not None:
            self.add_operations(measurements, are_measurement=True)
        if resets is not None:
            self.add_operations(resets, are_measurement=False)

    def add_operation(
        self, op: CollapsingOperationLocation, is_measurement: bool
    ) -> None:
        self._map.add_node(op, bipatite=int(is_measurement))

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
            self._map.add_edge(measurement, reset)

        if stabiliser is not None:
            self._add_stabiliser(measurement, reset_time_coordinate, stabiliser)

    @property
    def interaction_graph(self) -> nx.Graph:
        return self._map

    @property
    def stabilisers(self) -> dict[CollapsingOperationLocation, dict[int, Stabiliser]]:
        return self._stabilisers
