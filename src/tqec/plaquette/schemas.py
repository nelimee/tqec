from __future__ import annotations

from pydantic import BaseModel

from tqec.circuit.schemas import ScheduledCircuitModel, SupportedCircuitTypeEnum
from tqec.plaquette.plaquette import Plaquette
from tqec.plaquette.qubit import PlaquetteQubits


class PlaquetteModel(BaseModel):
    qubits: PlaquetteQubits
    circuit: ScheduledCircuitModel

    def to_plaquette(self) -> Plaquette:
        return Plaquette(self.qubits, self.circuit.to_scheduled_circuit())

    @staticmethod
    def from_plaquette(
        plaquette: Plaquette, circuit_type: SupportedCircuitTypeEnum
    ) -> PlaquetteModel:
        return PlaquetteModel(
            qubits=plaquette.qubits,
            circuit=ScheduledCircuitModel.from_scheduled_circuit(
                plaquette.circuit, circuit_type
            ),
        )
