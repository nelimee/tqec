from __future__ import annotations

import networkx as nx
from pydantic import BaseModel

from tqec.enums import CornerPositionEnum


class RelativePositionModel(BaseModel):
    start_idx: int
    end_idx: int
    start_corner: CornerPositionEnum
    end_corner: CornerPositionEnum


class RelativePositionsModel(BaseModel):
    positions: list[RelativePositionModel]

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for relpos in self.positions:
            G.add_edge(
                relpos.start_idx,
                relpos.end_idx,
                relative_position=(relpos.start_corner, relpos.end_corner),
            )
        return G

    @staticmethod
    def from_networkx(G: nx.DiGraph) -> RelativePositionsModel:
        return RelativePositionsModel(
            positions=[
                RelativePositionModel(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_corner=start_corner,
                    end_corner=end_corner,
                )
                for start_idx, end_idx, (
                    start_corner,
                    end_corner,
                ) in G.edges.data(
                    "relative_position"  # type: ignore
                )
            ]
        )
