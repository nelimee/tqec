"""ZX graph representation of a 3D spacetime defect diagram."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, cast
from enum import Enum
from dataclasses import dataclass, astuple
import itertools

import networkx as nx
import numpy as np
import numpy.typing as npt

from tqec.direction import Direction3D
from tqec.position import Position3D
from tqec.exceptions import TQECException

if TYPE_CHECKING:
    from tqec.sketchup.block_graph import BlockGraph


class NodeType(Enum):
    """Valid node types in a ZX graph."""

    X = "x"  # X-type node
    Z = "z"  # Z-type node
    V = "v"  # Virtual node that represents an open port

    def dual(self) -> NodeType:
        if self == NodeType.X:
            return NodeType.Z
        elif self == NodeType.Z:
            return NodeType.X
        else:
            return self


@dataclass(frozen=True)
class ZXNode:
    position: Position3D
    node_type: NodeType

    @property
    def is_virtual(self) -> bool:
        """Check if the node is virtual."""
        return self.node_type == NodeType.V


@dataclass(frozen=True)
class ZXEdge:
    u: ZXNode
    v: ZXNode
    has_hadamard: bool = False

    def __post_init__(self) -> None:
        if not self.u.position.is_neighbour(self.v.position):
            raise TQECException("An edge must connect two nearby nodes.")
        # Ensure position of u is less than v
        u, v = self.u, self.v
        if self.u.position > self.v.position:
            object.__setattr__(self, "u", v)
            object.__setattr__(self, "v", u)

    @property
    def direction(self) -> Direction3D:
        """Get the direction of the edge."""
        u, v = self.u.position, self.v.position
        if u.x != v.x:
            return Direction3D.X
        if u.y != v.y:
            return Direction3D.Y
        return Direction3D.Z


_NODE_DATA_KEY = "tqec_zx_node_data"
_EDGE_DATA_KEY = "tqec_zx_edge_data"

CORRELATION_COLOR: dict[NodeType, str] = {
    NodeType.X: "#ff0000",
    NodeType.Z: "#0000ff",
}


class ZXGraph:
    def __init__(self, name: str) -> None:
        """An undirected graph representation of a 3D spacetime defect diagram.

        Despite the name, the graph is not exactly the ZX-calculus graph
        as rewrite rules can not be applied to the graph arbitrarily.
        The graph must correspond to a valid 3D spacetime diagram, which
        can be realized with the lattice surgery on the 2D patches of
        surface code. And rewrite rules can only be applied with respect
        to a valid physical realization of the spacetime diagram.

        Note that not all ZX graph admits a valid spacetime diagram
        representation. And the graph construction **does not check**
        the validity constraints.
        """
        self._name = name
        # Internal undirected graph representation
        self._graph = nx.Graph()

    @property
    def name(self) -> str:
        """The name of the graph."""
        return self._name

    @property
    def nx_graph(self) -> nx.Graph:
        """The internal networkx graph representation."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the graph."""
        return cast(int, self._graph.number_of_nodes())

    @property
    def num_edges(self) -> int:
        """The number of edges in the graph."""
        return cast(int, self._graph.number_of_edges())

    @property
    def nodes(self) -> list[ZXNode]:
        """Return a list of nodes in the graph."""
        return [data[_NODE_DATA_KEY] for _, data in self._graph.nodes(data=True)]

    @property
    def edges(self) -> list[ZXEdge]:
        """Return a list of edges in the graph."""
        return [data[_EDGE_DATA_KEY] for _, _, data in self._graph.edges(data=True)]

    @property
    def leaf_nodes(self) -> list[ZXNode]:
        """Get the leaf nodes of the graph."""
        return [node for node in self.nodes if self._graph.degree(node.position) == 1]

    @property
    def isolated_nodes(self) -> list[ZXNode]:
        """Get the isolated nodes of the graph."""
        return [node for node in self.nodes if self._graph.degree(node.position) == 0]

    def add_node(
        self,
        position: Position3D,
        node_type: NodeType,
        raise_if_exist: bool = True,
    ) -> None:
        """Add a node to the graph.

        Args:
            position: The 3D position of the node.
            node_type: The type of the node.
            raise_if_exist: Whether to raise an exception if the position already exists
                in the graph. If set to False, when the position already exists, the node
                type will be updated to the new type. Default is True.
        """
        if raise_if_exist and position in self._graph:
            raise TQECException(f"Node {position} already exists in the graph.")
        self._graph.add_node(position, **{_NODE_DATA_KEY: ZXNode(position, node_type)})

    def add_z_node(self, position: Position3D) -> None:
        """Add a Z-type node to the graph."""
        self.add_node(position, NodeType.Z)

    def add_x_node(self, position: Position3D) -> None:
        """Add an X-type node to the graph."""
        self.add_node(position, NodeType.X)

    def add_virtual_node(self, position: Position3D) -> None:
        """Add a virtual node to the graph."""
        self.add_node(position, NodeType.V)

    def add_edge(
        self,
        u: Position3D,
        v: Position3D,
        has_hadamard: bool = False,
    ) -> None:
        """Add an edge to the graph.

        Args:
            u: The position of the first node.
            v: The position of the second node.
            has_hadamard: Whether the edge has a Hadamard transition.
        """
        if u not in self._graph or v not in self._graph:
            raise TQECException("Both nodes must exist in the graph.")
        u_node: ZXNode = self._graph.nodes[u][_NODE_DATA_KEY]
        v_node: ZXNode = self._graph.nodes[v][_NODE_DATA_KEY]
        self._graph.add_edge(
            u, v, **{_EDGE_DATA_KEY: ZXEdge(u_node, v_node, has_hadamard)}
        )

    def get_node(self, position: Position3D) -> ZXNode | None:
        """Get the node by position."""
        if position not in self._graph:
            return None
        return cast(ZXNode, self._graph.nodes[position][_NODE_DATA_KEY])

    def get_edge(self, u: Position3D, v: Position3D) -> ZXEdge | None:
        """Get the edge by its endpoint positions."""
        if not self._graph.has_edge(u, v):
            return None
        return cast(ZXEdge, self._graph.edges[u, v][_EDGE_DATA_KEY])

    def edges_at(self, position: Position3D) -> list[ZXEdge]:
        """Get the edges incident to a node."""
        return [
            data[_EDGE_DATA_KEY]
            for _, _, data in self._graph.edges(position, data=True)
        ]

    def draw(
        self,
        *,
        show_correlation_subgraph_index: int | None = None,
        figsize: tuple[float, float] = (5, 6),
        title: str | None = None,
        node_size: int = 400,
        hadamard_size: int = 200,
        edge_width: int = 1,
        correlation_edge_width: int = 3,
    ) -> None:
        """Draw the 3D graph using matplotlib.

        Args:
            show_correlation_subgraph_index: The index of the correlation subgraph to show.
            figsize: The figure size. Default is (5, 6).
            title: The title of the plot. Default is the name of the graph.
            node_size: The size of the node. Default is 400.
            hadamard_size: The size of the Hadamard transition. Default is 200.
            edge_width: The width of the edge. Default is 1.
            correlation_edge_width: The width of the correlation edge. Default is 3.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        correlation_subgraph: ZXGraph | None = None
        if show_correlation_subgraph_index is not None:
            correlation_subgraphs = self.find_correlation_subgraphs()
            if show_correlation_subgraph_index >= len(correlation_subgraphs):
                raise TQECException(
                    f"Only {len(correlation_subgraphs)} correlation subgraphs found."
                    f"Index {show_correlation_subgraph_index} is out of range."
                )
            correlation_subgraph = correlation_subgraphs[
                show_correlation_subgraph_index
            ]

        fig = plt.figure(figsize=figsize)
        # See https://matplotlib.org/stable/users/explain/toolkits/mplot3d.html
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

        def positions_array(*positions: Position3D) -> npt.NDArray[np.int_]:
            return np.array([astuple(p) for p in positions]).T

        non_virtual_nodes = [n for n in self.nodes if not n.is_virtual]
        non_virtual_nodes_array = positions_array(
            *[n.position for n in non_virtual_nodes]
        )
        if non_virtual_nodes_array.size > 0:
            ax.scatter(
                *non_virtual_nodes_array,
                s=node_size,
                c=[
                    "black" if n.node_type == NodeType.X else "white"
                    for n in non_virtual_nodes
                ],
                alpha=1.0,
                edgecolors="black",
            )

        for edge in self.edges:
            pos_array = positions_array(edge.u.position, edge.v.position)
            ax.plot(
                *pos_array,
                color="tab:gray",
                linewidth=edge_width,
            )
            if edge.has_hadamard:
                hadamard_position = np.mean(pos_array, axis=1)
                # use yellow square to indicate Hadamard transition
                ax.scatter(
                    *hadamard_position,
                    s=hadamard_size,
                    c="yellow",
                    alpha=1.0,
                    edgecolors="black",
                    marker="s",
                )

        if correlation_subgraph is not None:
            for edge in correlation_subgraph.edges:
                pos_array = positions_array(edge.u.position, edge.v.position)
                if not edge.has_hadamard:
                    correlation_type = edge.u.node_type
                    ax.plot(
                        *pos_array,
                        color=CORRELATION_COLOR[correlation_type],
                        linewidth=correlation_edge_width,
                    )
                else:
                    hadamard_position = np.mean(pos_array, axis=1)
                    for node in [edge.u, edge.v]:
                        ax.plot(
                            *np.hstack(
                                [
                                    hadamard_position.reshape(3, 1),
                                    positions_array(node.position),
                                ]
                            ),
                            color=CORRELATION_COLOR[node.node_type],
                            linewidth=correlation_edge_width,
                        )

        ax.grid(False)
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

        plot_radius = 0.5 * max(
            abs(limits[1] - limits[0]) for limits in [x_limits, y_limits, z_limits]
        )

        ax.set_xlim3d(
            [np.mean(x_limits) - plot_radius, np.mean(x_limits) + plot_radius]
        )
        ax.set_ylim3d(
            [np.mean(y_limits) - plot_radius, np.mean(y_limits) + plot_radius]
        )
        ax.set_zlim3d(
            [np.mean(z_limits) - plot_radius, np.mean(z_limits) + plot_radius]
        )
        ax.set_title(title or self.name)
        fig.tight_layout()
        plt.show()

    def to_block_graph(self, name: str = "") -> BlockGraph:
        """Construct a block graph from a ZX graph.

        The ZX graph includes the minimal information required to construct the block graph,
        but not guaranteed to admit a valid block structure. The block structure will be inferred
        from the ZX graph and validated.

        Args:
            name: The name of the new block graph.

        Returns:
            The constructed block graph.
        """
        from tqec.sketchup.block_graph import BlockGraph

        return BlockGraph.from_zx_graph(self, name=name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZXGraph):
            return False
        return cast(bool, nx.utils.graphs_equal(self._graph, other._graph))

    def _find_correlation_subgraphs_dfs(
        self,
        parent_corr_node: ZXNode,
        parent_zx_node_type: NodeType,
        visited_positions: set[Position3D],
    ) -> list[set[ZXEdge]]:
        """Recursively find all the correlation subgraphs starting from a node,
        represented by the correlation edges in the subgraph.

        The algorithm is as follows:
        1. Initialization
            - Initialize the `correlation_subgraphs` as an empty list.
            - Add the parent to the `visited_positions`.
            - Initialize a list `branched_subgraph` to hold subgraphs constructed
            from the neighbors of the parent.
        2. Iterate through all edges connected to parent. For each edge:
            - If the child node is already visited, skip the edge.
            - Determine the correlation type of the child node based on the correlation
            type of the parent and the Hadamard transition.
            - Create a new correlation node for the child and the correlation edge
            between the parent and the child.
            - Recursively call this method to find the correlation subgraphs starting
            from the child. Then add the edge in the last step to each of the subgraphs.
            Append the subgraphs to the `branched_subgraph`.
        3. Post-processing
            - If no subgraphs are found, return a single empty subgraph.
            - If the color of the node matches the correlation type, all the children
            should be traversed. Iterate through all the combinations where exactly one
            subgraph is selected from each child, and union them to form a new subgraph.
            Append the new subgraph to the `correlation_subgraphs`.
            - If the color of the node does not match the correlation type, only one
            child can be traversed. Append all the subgraphs in the `branched_subgraph`
            to the `correlation_subgraphs`.
        """
        correlation_subgraphs: list[set[ZXEdge]] = []
        parent_position = parent_corr_node.position
        parent_corr_type = parent_corr_node.node_type
        visited_positions.add(parent_position)

        branched_subgraphs: list[list[set[ZXEdge]]] = []
        for edge in self.edges_at(parent_position):
            cur_zx_node = edge.u if edge.v.position == parent_position else edge.v
            if cur_zx_node.position in visited_positions:
                continue
            cur_corr_type = (
                parent_corr_type.dual() if edge.has_hadamard else parent_corr_type
            )
            cur_corr_node = ZXNode(cur_zx_node.position, cur_corr_type)
            edge_between_cur_parent = ZXEdge(
                parent_corr_node, cur_corr_node, edge.has_hadamard
            )
            branched_subgraphs.append(
                [
                    subgraph | {edge_between_cur_parent}
                    for subgraph in self._find_correlation_subgraphs_dfs(
                        cur_corr_node,
                        cur_zx_node.node_type,
                        copy(visited_positions),
                    )
                ]
            )
        if not branched_subgraphs:
            return [set()]
        # the color of node matches the correlation type
        # broadcast the correlation type to all the neighbors
        if parent_zx_node_type == parent_corr_type:
            for prod in itertools.product(*branched_subgraphs):
                correlation_subgraphs.append(set(itertools.chain(*prod)))
            return correlation_subgraphs

        # the color of node does not match the correlation type
        # only one of the neighbors can be the correlation path
        correlation_subgraphs.extend(itertools.chain(*branched_subgraphs))
        return correlation_subgraphs

    def find_correlation_subgraphs(self) -> list[ZXGraph]:
        """Find the correlation subgraphs of the ZX graph.

        Here a correlation subgraph is defined as a subgraph of the `ZXGraph`
        that represents the correlation surface within a 3D spacetime diagram.
        Each node in the correlation subgraph is composed of its position and
        the correlation surface type, which is either `NodeType.X` or `NodeType.Z`.

        For the closed diagram, the correlation subgraph represents the correlation
        between the measured logical observables. For the open diagram, the
        correlation subgraph represents the correlation between the measured logical
        observables and the input/output observables, which can be combined with
        the expected stabilizer flow to verify the correctness of the computation.

        A recursive depth-first search algorithm is used to find the correlation
        subgraphs starting from each leaf node. The algorithm is described in the
        method `_find_correlation_subgraphs_dfs`.
        """
        single_node_correlation_subgraphs: list[ZXGraph] = []
        multi_edges_correlation_subgraphs: dict[frozenset[ZXEdge], ZXGraph] = {}
        num_subgraphs = 0
        for node in self.isolated_nodes:
            if node.is_virtual:
                continue
            subgraph = ZXGraph(f"Correlation subgraph {num_subgraphs}")
            subgraph.add_node(node.position, node.node_type)
            single_node_correlation_subgraphs.append(subgraph)
            num_subgraphs += 1

        def add_subgraphs(node: ZXNode, correlation_type: NodeType) -> None:
            nonlocal num_subgraphs
            root_corr_node = ZXNode(node.position, correlation_type)
            for edges in self._find_correlation_subgraphs_dfs(
                root_corr_node, node.node_type, set()
            ):
                if frozenset(edges) in multi_edges_correlation_subgraphs:
                    continue
                subgraph = ZXGraph(
                    f"Correlation subgraph {num_subgraphs} of {self.name}"
                )
                for edge in edges:
                    subgraph.add_node(
                        edge.u.position, edge.u.node_type, raise_if_exist=False
                    )
                    subgraph.add_node(
                        edge.v.position, edge.v.node_type, raise_if_exist=False
                    )
                    subgraph.add_edge(
                        edge.u.position, edge.v.position, edge.has_hadamard
                    )
                multi_edges_correlation_subgraphs[frozenset(edges)] = subgraph
                num_subgraphs += 1

        for node in self.leaf_nodes:
            if node.is_virtual:
                for correlation_type in [NodeType.X, NodeType.Z]:
                    add_subgraphs(node, correlation_type)
            else:
                add_subgraphs(node, node.node_type)
        return single_node_correlation_subgraphs + list(
            multi_edges_correlation_subgraphs.values()
        )
