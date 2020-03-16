# from graph import Graphs

class PushRelabel: 
    def _get_active_node(self, graph, s, t):
        """
        :param s: source node
        :param t: target node
        :param graph: graph on which algo is applied
        :return: an active node in the graph that we can look at 
        """
        # Look at all nodes in the graph
        for node in graph.nodes():
            # Return node whose excess flow is greater than 0
            if not node is t and not node is s and node.overrun > 0:
                return node
        return None

    def _has_active_node(self, graph, s, t):
        """
        :param s: source node
        :param t: target node
        :param graph: graph on which algo is applied
        """
        return True if not _get_active_node(graph, s, t) is None else False

    def _push(self, node):
        """
        :param node: node that we consider
        Push load away from the current node if possible.
        If a neighboring node which is "closer" to the target can accept more load
        push it to the node. If no such node is found push fails meaning a relabel has
        to be executed.
        """
        success = False

        # Look at all outgoing edges of that node
        for edge in node.outgoing_edges():
            # Find a corresponding neighbor
            neighbor = edge.destination()
            # Conditions to make algo applicable
            if not node.dist == neighbor.dist + 1 or edge.load == edge.capacity:
                continue
            success = True
            # Create undirected edge
            reverse_edge = node.edge_from(neighbor)
            
            # Define push, flow, and excess flow (overrun) 
            push = min(edge.capacity - edge.load, node.overrun)
            edge.load         += push
            reverse_edge.load -= push
            neighbor.overrun  += push
            node.overrun      -= push

            # Check output of push step 
            print("[*] pushing %i from %s to %s" % (push, node.name(), neighbor.name()))

            # If the node does not overrun, we are done with it
            if node.overrun == 0:
                break

        return success

    def _relabel(self, node):
        """
        :param node: node that we consider
        Relabel a node.
        Adjusts the dist value of the current node to the minimun dist
        value of its neighbors plus one.
        """
        # Use to redefine label of the node
        min_dist = None

        # Look at all incident edges to that node
        for edge in node.outgoing_edges():
            # If this edge is saturated, discard
            if edge.load == edge.capacity:
                continue
            # If excess flow > 0 and input node label <= output node label
            if min_dist is None or edge.destination().dist < min_dist:
                min_dist = edge.destination().dist

        # Relabel node 
        node.dist = min_dist + 1


    def solve_max_flow(self, graph, s, t):
        """
        Solves the max flow prolem using the push-relabel algorithm for the given
        graph and source/target node.
        """
        #
        # initialize algorithm specific data
        #
        for node in graph.nodes():
            node.dist = 0
            node.overrun = 0
        for edge in graph.edges():
            edge.load = 0
            # add reverse edges
            if not graph.has_reverse_edge(edge):
                graph.add_edge(edge.destination(), edge.source(), {"capacity" : 0, "load" : 0, "tmp" : True})
        # Initialize source node label (to total number of vertices in the graph)
        s.dist = len(graph.nodes())
        # populate edges going out of the source node
        for edge in s.outgoing_edges():
            edge.load = edge.capacity
            edge.destination().overrun = edge.load
            edge.destination().edge_to(s).load = -edge.capacity

        #
        # solve the max flow problem
        #
        # While there is an applicable push or relabel, do
        while _has_active_node(graph, s, t):
            node = _get_active_node(graph, s, t)
            if not _push(node):
                _relabel(node)
                print("[*] relabeling %s to dist %i" % (node.name(), node.dist))

        # cleanup
        for edge in graph.edges():
            if hasattr(edge, "tmp"):
                graph.remove_edge(edge)