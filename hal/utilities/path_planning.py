"""path_planning.py: A collection of utilities for use in path planning.

This module contains a collection of utilities that can be used to simplify
path planning for autonomous mobile robots and robotic manipulators.
"""
import numpy as np
import heapq
import pal.utilities.math as qm


def hermite_position(s, p1, p2, t1, t2):
    """Compute the position at point 's' along a cubic Hermite spline.

    Args:
        s (float): Parameter in the range of [0, 1]; distance along the spline.
        p1 (numpy.ndarray): Start point (2x1 array) of the spline.
        p2 (numpy.ndarray): End point (2x1 array) of the spline.
        t1 (numpy.ndarray): Tangent at the start point (2x1 array).
        t2 (numpy.ndarray): Tangent at the end point (2x1 array).

    Returns:
        numpy.ndarray: Position of the spline at parameter s (2x1 array).
    """
    return (
        (2 * s**3 - 3 * s**2 + 1) * p1
        + (s**3 - 2 * s**2 + s) * t1
        + (-2 * s**3 + 3 * s**2) * p2
        + (s**3 - s**2) * t2
    )

def hermite_tangent(s, p1, p2, t1, t2):
    """Compute the derivative of a cubic Hermite spline at s.

    Args:
        s (float): Parameter in the range of [0, 1]; distance along the spline.
        p1 (numpy.ndarray): Start point (2x1 array) of the spline.
        p2 (numpy.ndarray): End point (2x1 array) of the spline.
        t1 (numpy.ndarray): Tangent at the start point (2x1 array).
        t2 (numpy.ndarray): Tangent at the end point (2x1 array).

    Returns:
        numpy.ndarray: Tangent of the spline at parameter s (2x1 array).
    """
    return (
        (6*s**2 - 6*s) * p1
        + (-6*s**2 + 6*s) * t1
        + (3*s**2 - 4*s + 1) * p2
        + (3*s**2 - 2*s) * t2
    )

def hermite_heading(s, p1, p2, t1, t2):
    """Compute the heading angle at s along a cubic Hermite spline.

    Args:
        s (float): Parameter in the range of [0, 1]; distance along the spline.
        p1 (numpy.ndarray): Start point (2x1 array) of the spline.
        p2 (numpy.ndarray): End point (2x1 array) of the spline.
        t1 (numpy.ndarray): Tangent at the start point (2x1 array).
        t2 (numpy.ndarray): Tangent at the end point (2x1 array).

    Returns:
        float: Heading angle of the spline at s (in radians).
    """
    t = hermite_tangent(s, p1, p2, t1, t2)
    return np.arctan2(t[1,0], t[0,0])

def SCSPath(startPose, endPose, radius, stepSize = 0.1):
    """Calculate the path between two poses using at most one turn.

    Args:
        startPose (numpy.ndarray): Starting pose in the form [x, y, th].
        endPose (numpy.ndarray): Ending pose in the form [x, y, th].
        radius (float): Radius of the turn connecting the two poses.
        stepSize (float, optional): Distance between points. Defaults to 0.1

    Returns:
        A tuple containing the path as a 2xn numpy.ndarray and the path length.
    """
    # Extract positions and headings from poses [x; y; th]
    p1 = startPose[:2, :]
    th1 = startPose[2, 0]
    p2 = endPose[:2, :]
    th2 = endPose[2, 0]

    # Find tangent vectors at p1 and p2
    t1 = np.array([[np.cos(th1)], [np.sin(th1)]])
    t2 = np.array([[np.cos(th2)], [np.sin(th2)]])

    # Determine turn direction (CCW = 1, CW = -1)
    dir = 1 if qm.signed_angle(t1, p2-p1) > 0 else -1

    # Calculate normals
    n1 = radius * np.array([[-t1[1,0]], [t1[0,0]]]) * dir
    n2 = radius * np.array([[-t2[1,0]], [t2[0,0]]]) * dir

    # = Calculate 'c': centre point for circle
    tol = 0.01
    if np.abs(qm.wrap_to_pi(th2 - th1)) < tol:
        v = p2 - p1
        v_uv = v / np.linalg.norm(v)
        if 1 - np.abs(np.dot(t1.squeeze(), v_uv.squeeze())) < tol:
            c = p2 + n1
        else:
            return None, None
    elif np.abs(qm.wrap_to_pi(th2 - th1 + np.pi)) < tol:
        v = (p2 + 2*n2) - p1
        v_uv = v / np.linalg.norm(v)
        if 1 - np.abs(np.dot(t1.squeeze(), v_uv.squeeze())) < tol:
            s = np.dot(t1.squeeze(), v.squeeze())
            if s < tol:
                c = p1 + n1
            else:
                c = p2 + n2
        else:
            return None, None

    else:
        # Calculate points used for finding c
        d1 = p1 + n1
        d2 = p2 + n2

        # Solve system of linear equations to find intersection point
        # (alpha and beta are the distances along each ray)
        A = np.hstack((t1, -t2))
        b = d2 - d1

        # Check if rays are parallel
        alpha, beta = np.linalg.solve(A, b)
        if alpha >= -tol and beta <= tol:
            c = d1 + alpha * t1
        else:
            return None, None

    # Calculate b1, b2
    b1 = c - n1
    b2 = c - n2

    # Discretize line-segment p1 -> b1
    line1 = np.empty((2,0))
    line1_length = np.linalg.norm(b1 - p1)

    if line1_length > stepSize:
        ds = (1.0 / line1_length) * stepSize
        s = ds
        while s < 1:
            p = p1 + s * (b1 - p1)
            line1 = np.hstack((line1, p))
            s += ds

    # Discretize arc b1 -> b2 with centerpoint c
    arc = np.empty((2,0))
    ang_dist = qm.wrap_to_2pi(dir*qm.signed_angle(b1-c, b2-c))
    arc_length = np.abs(ang_dist * radius)

    if arc_length > stepSize:
        start_angle = np.arctan2(b1[1] - c[1], b1[0] - c[0])
        dth = (2*np.pi / (np.pi*2*radius)) * stepSize

        s = dth
        while s < ang_dist:
            th = start_angle + s * dir
            p = c + np.array([np.cos(th), np.sin(th)]) * radius
            arc = np.hstack((arc, p))
            s += dth

    # Discretize line-segment b2 -> p2
    line2 = np.empty((2,0))
    line2_length = np.linalg.norm(b2 - p2)

    if line2_length > stepSize:
        ds = (1.0 / line2_length) * stepSize
        s = ds
        while s < 1:
            p = b2 + s * (p2 - b2)
            line2 = np.hstack((line2, p))
            s += ds

    # Stack three sections horizontally and return the result
    path = np.hstack((line1, arc, line2))

    # Calculate the length of the path
    path_length = line1_length + arc_length + line2_length

    return path, path_length


class RoadMapNode:
    """Class for representing nodes in the graph of a RoadMap

    Attributes:
        pose (numpy.ndarray): Node's pose in the form [x, y, th].
        inEdges (list): List of incoming edges.
        outEdges (list): List of outgoing edges.
    """

    def __init__(self, pose):
        """Initialize a RoadMapNode instance.

        Args:
            pose (list or numpy.ndarray): Node's pose in the form [x, y, th].
        """
        assert len(pose) == 3, "Pose must be in the form of [x, y, th]"

        self.pose = np.array(pose).reshape(3, 1)
        self.inEdges = []
        self.outEdges = []

class RoadMapEdge:
    """Class for representing edges in the graph of a RoadMap.

    Attributes:
        fromNode (RoadMapNode): Starting node of the edge.
        toNode (RoadMapNode): Ending node of the edge.
        waypoints (numpy.ndarray): Waypoints along the edge.
        length (float): Length of the edge.
    """
    def __init__(self, fromNode, toNode):
        """Initialize a RoadMapEdge instance.

        Args:
            fromNode (RoadMapNode): Starting node of the edge.
            toNode (RoadMapNode): Ending node of the edge.
        """
        assert (isinstance(fromNode, RoadMapNode)
            and isinstance(toNode, RoadMapNode)), \
            "Both fromNode and toNode must be instances of RoadMapNode"

        self.fromNode = fromNode
        self.toNode = toNode
        self.waypoints = None
        self.length = None

class RoadMap:
    """
    Graph-based roadmap for generating paths between points in a road network.

    Attributes:
        nodes (list): List of nodes in the roadmap.
        edges (list): List of edges in the roadmap.
    """

    def __init__(self):
        """Initialize a RoadMap instance."""
        self.nodes = []
        self.edges = []

    def add_node(self, pose):
        """
        Add a node to the roadmap.

        Args:
            pose (list or numpy.ndarray): Node's pose in the form [x, y, th].
        """
        self.nodes.append(RoadMapNode(pose))

    def add_edge(self, fromNode, toNode, radius):
        """Add an edge between two nodes in the roadmap.

        Args:
            fromNode (int or RoadMapNode): Starting node (index or instance).
            toNode (int or RoadMapNode): Ending node (index or instance).
            radius (float): Radius of the turn connecting the two nodes.
        """
        if type(fromNode) == int:
            fromNode = self.nodes[fromNode]
        if type(toNode) == int:
            toNode = self.nodes[toNode]

        edge = RoadMapEdge(fromNode, toNode)
        self.edges.append(edge)
        fromNode.outEdges.append(edge)
        toNode.inEdges.append(edge)

        self._calculate_trajectory(edge, radius)

    def remove_edge(self, fromNode, toNode):
        """Remove an edge between two nodes in the roadmap.

        Args:
            fromNode (int or RoadMapNode): Starting node (index or instance).
            toNode (int or RoadMapNode): Ending node (index or instance).
        """
        if type(fromNode) == int:
            fromNode = self.nodes[fromNode]
        if type(toNode) == int:
            toNode = self.nodes[toNode]

        for edge in self.edges:
            if edge.fromNode == fromNode and edge.toNode == toNode:
                self.edges.remove(edge)
                fromNode.outEdges.remove(edge)
                toNode.inEdges.remove(edge)
                return

    def _calculate_trajectory(self, edge, radius):
        """Calculate the waypoints and length of the given edge

        Args:
            edge (RoadMapEdge): Instance of the edge.
            radius (float): Radius of the turn connecting the two nodes.
        """
        points, length = SCSPath(
            startPose=edge.fromNode.pose,
            endPose=edge.toNode.pose,
            radius=radius,
            stepSize=0.01
        )
        edge.waypoints = points
        edge.length = length

    def get_node_pose(self, nodeID):
        """Get the pose of a node by its index.

        Args:
            nodeID (int): Index of the node in the roadmap.

        Returns:
            numpy.ndarray: Pose of the node in the form [x, y, th].
        """
        return self.nodes[nodeID].pose

    def generate_path(self, nodeSequence):
        """
        Generate the shortest path passing through the given sequence of nodes

        Args:
            nodeSequence (list or tuple): Sequence of node indices.

        Returns:
            numpy.ndarray: generated path as a 2xn numpy array
        """
        assert isinstance(nodeSequence, (list, tuple)), \
            "Node sequence must be provided as either a list or a tuple."
        
        path = np.empty((2, 0))
        for i in range(len(nodeSequence) - 1):
            pathSegment = self.find_shortest_path(
                nodeSequence[i],
                nodeSequence[i+1]
            )
            if pathSegment is None:
                return None
            path = np.hstack((path, pathSegment[:, :-1]))
        return path

    def _heuristic(self, node, goalNode):
        """Calculate the heuristic cost between two nodes.

        Args:
            node (RoadMapNode): Instance of the current node.
            goalNode (RoadMapNode): Instance of the goal node.

        Returns:
            float: Heuristic cost between the current node and goal node.
        """
        return np.linalg.norm(goalNode.pose[:2, :] - node.pose[:2, :])

    def find_shortest_path(self, startNode, goalNode):
        """Find the shortest path between two nodes using the A* algorithm.

        Args:
            startNode (int or RoadMapNode): Starting node (index or instance).
            goalNode (int or RoadMapNode): Goal node (index or instance).
            radius (float): Minimum turning radius.

        Returns:
            path: generated path as a 2xn numpy array
        """
        if startNode == goalNode:
            return None

        if type(startNode) == int:
            startNode = self.nodes[startNode]
        if type(goalNode) == int:
            goalNode = self.nodes[goalNode]

        # Initialize the open set and closed set
        openSet = []
        closedSet = set()

        # Add the start node to the open set with a cost of 0 and an
        # f-score equal to the heuristic estimate
        heapq.heappush(
            openSet,
            (0 + self._heuristic(startNode, goalNode), startNode)
        )

        # Initialize the g-scores for each node to infinity
        gScore = {node: float('inf') for node in self.nodes}
        gScore[startNode] = 0

        # Initialize the 'came from' (node, edge) pair
        # for each node to None
        cameFrom = {node: None for node in self.nodes}

        while openSet:
            # Pop the node with the lowest f-score from the open set
            currentNode = heapq.heappop(openSet)[1]

            if currentNode == goalNode:
                # Goal node found, construct the optimal path, then return
                path = goalNode.pose[:2,:]
                node = goalNode
                while True:
                    (node, edge) = cameFrom[node]
                    path = np.hstack((
                        node.pose[:2,:],
                        edge.waypoints,
                        path
                    ))
                    if cameFrom[node] is None:
                        break
                return path

            closedSet.add(currentNode)

            for edge in currentNode.outEdges:
                neighborNode = edge.toNode
                if neighborNode in closedSet:
                    # Neighbor node already explored
                    continue

                if edge.length is None:
                    tentative_g = float('inf')
                else:
                    tentative_g = gScore[currentNode] + edge.length

                if tentative_g < gScore[neighborNode]:
                    # New path to neighbor node found,
                    # update parent pointer and g-score
                    cameFrom[neighborNode] = (currentNode, edge)
                    gScore[neighborNode] = tentative_g

                    # Add the neighbor node to the open set with a cost
                    # equal to the g-score plus the heuristic estimate
                    hScore = self._heuristic(neighborNode, goalNode)
                    heapq.heappush(
                        openSet,
                        (gScore[neighborNode] + hScore, neighborNode)
                    )

        # Open set is empty and goal node not found, no path exists
        return None

    def display(self, ax=None):
        """Display the roadmap with nodes, edges, and labels using matplotlib.

        Args:
            ax (matplotlib.axes.Axes, optional): A matplotlib axes object
                to draw the roadmap on. If not provided, a new figure and axes
                will be created.

        Returns:
            tuple: A tuple containing the matplotlib.pyplot instance
                and the axes object.
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import matplotlib.patheffects as path_effects

        if ax is None:
            fig, ax = plt.subplots()
            fig.gca().set_aspect('equal', adjustable='box')

        for edge in self.edges:
            if edge.waypoints is None:
                continue
            ax.plot(
                edge.waypoints[0, :],
                edge.waypoints[1, :],
                'green',
                linestyle='-',
                linewidth=1.5,
                zorder=1
            )

        for idx, node in enumerate(self.nodes):
            x, y, heading = node.pose[:, 0]
            ax.plot(x, y, marker='o', markersize=6, color='red')

            text = ax.text(
                x + 0.03,
                y + 0.03,
                str(idx),
                fontsize=12,
                color='white'
            )
            text.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()
            ])

            # Draw arrow representing the heading direction
            arrow_length = 0.1
            arrow_dx = arrow_length * np.cos(heading)
            arrow_dy = arrow_length * np.sin(heading)
            ax.arrow(
                x,
                y,
                arrow_dx,
                arrow_dy,
                head_width=0.05,
                head_length=0.1,
                fc='red',
                ec='red'
            )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        return plt, ax