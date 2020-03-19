#push-relabel algorithm

class PushRelabel:
    def __init__(self):
        self.capacity = None  # capacity matrix
        self.max_flow = 0  # value of maximum flow
        self.mincut = None  # min cut

    def capacity_matrix(self, w_if, w_ib, hori_w_if, vert_w_if):
        """
        :param w_if: terminal weights associated with source node
        :param w_ib: terminal weights associated with target node
        :param hori_w_if: horizontal non terminal weights
        :param vert_w_if: vertical non terminal weights
        :return: equivalent of capacity matrix but as a dictionnary
        With key is origin node, value is dictionnary.
        With key as destination node and value is the weight of the corresponding edge
        """
        dico = {}
        sub_dico = {}
        w = w_if.shape[0]
        h = w_if.shape[1]
        target_num = w * h + 1

        # Source and target node
        for y in range(h):
            for x in range(w):
                node_id = y * h + x + 1
                sub_dico[node_id] = w_if[x, y]
                dico[node_id] = {target_num: w_ib[x, y]}
        dico[0] = sub_dico

        # Add horizontal non terminal weights
        for y in range(h):
            for x in range(w-1):
                node_id = y * h + x + 1
                dico[node_id][node_id + 1] = hori_w_if[x][y]

        # Add vertical non terminal weights
        for y in range(h-1):
            for x in range(w):
                node_id = y * h + x + 1
                dico[node_id][node_id + w] = vert_w_if[x][y]

        # Create matrix
        # n = h * w + 2
        # C = [[0] * n for i in range(n)]
        # for key in dico:
        #     for key2 in dico[key]:
        #         C[key][key2] = dico[key][key2]

        self.capacity = dico

    def MaxFlow(self, C, s, t):
        """
        :param C: capacity matrix of the graph
        :param s: source node
        :param t: target node
        :return: return the maximum flow value (what leaves from source node)
        """
        n = len(C) # C is the capacity matrix
        F = [[0] * n for i in range(n)] # Big flow matrix

        # the residual capacity from u to v is C[u][v] - F[u][v]
        height = [0] * n # height of node
        excess = [0] * n # flow into node minus flow from node
        seen   = [0] * n # neighbours seen since last relabel
        # node "queue"
        nodelist = [i for i in range(n) if i != s and i != t]

        def push(u, v):
            """
            :param u, v: two adjacent vertices of the graph, link by an edge
            Push the flow from a vertex with strictly postive overflow
            to one of its neighbours via a vertex with strictly positive residual capacity
            """
            send = min(excess[u], C[u][v] - F[u][v])
            F[u][v] += send
            F[v][u] -= send
            excess[u] -= send
            excess[v] += send

        def relabel(u):
            """
            :param u: node of the graph
            Find smallest new height for that node to make a push possible,
            if such a push is possible at all
            """
            # find smallest new height making a push possible,
            # if such a push is possible at all
            min_height = float('inf')
            for v in range(n):
                if C[u][v] - F[u][v] > 0:
                    min_height = min(min_height, height[v])
                    height[u] = min_height + 1

        def discharge(u):
            """
            :param u: targeted node of the graph
            Push or relabel process for node u
            """
            # If node has excess flow
            while excess[u] > 0:
                # Check neighbours
                if seen[u] < n:
                    v = seen[u]
                    # Push if conditions are met
                    if C[u][v] - F[u][v] > 0 and height[u] > height[v]:
                        push(u, v)
                    else:
                        seen[u] += 1
                # Relabel if we have check all neighbours of u and push is impossible
                else:
                    relabel(u)
                    seen[u] = 0

        # Set label of source to number of vertices
        height[s] = n

        # Saturate all edges incident to the source
        excess[s] = float("inf")
        for v in range(n):
            push(s, v)

        p = 0
        # Apply push relabel algo on the whole graph
        while p < len(nodelist):
            u = nodelist[p]
            old_height = height[u]
            # Push or relabel u
            discharge(u)
            # Move to front of the list if relabelling has been done
            if height[u] > old_height:
                nodelist.insert(0, nodelist.pop(p))
                p = 0
            else:
                p += 1

        self.max_flow = sum(F[s])



"""

# Make a capacity graph
# node s  o  p  q  r  t
C = [[ 0, 3, 3, 0, 0, 0 ],  # s
     [ 0, 0, 2, 3, 0, 0 ],  # o
     [ 0, 0, 0, 0, 2, 0 ],  # p
     [ 0, 0, 0, 0, 4, 2 ],  # q
     [ 0, 0, 0, 0, 0, 2 ],  # r
     [ 0, 0, 0, 0, 0, 3 ]]  # t

source = 0  # A
sink = 5   # F
max_flow_value = MaxFlow(C, source, sink)
print("Push-Relabeled(Preflow-push) algorithm")
print("max_flow_value is: ", max_flow_value)

"""
