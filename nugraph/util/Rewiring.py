from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge, to_undirected, remove_self_loops, contains_isolated_nodes, coalesce
from math import ceil
import torch

class Rewiring(BaseTransform):

    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: "pyg.data.HeteroData", rewire_type: str='maxdeg_unif') -> "pyg.data.HeteroData":
        for p in self.planes:
            n_nodes = data[p].num_nodes # Or edge_index.max() - edge_index.min() + 1 if the graph is in a batch

            ## Test 1: Change the target nodes, i.e. (i,j) -> (i,k)
            if rewire_type == 'permute1':
                data[p, 'plane', p].edge_index = Rewiring.permute_edges(data[p, 'plane', p].edge_index, num_nodes=n_nodes,
                                                                        algo=1)

            ## Test 2: Completely random set of edges, i.e. (i,j) -> (k,l)
            elif rewire_type == 'permute2':
                data[p, 'plane', p].edge_index = Rewiring.permute_edges(data[p, 'plane', p].edge_index, num_nodes=n_nodes,
                                                                        algo=2)
            ## Test 3: Edge dropout
            elif rewire_type == 'edge_dropout':
                data[p, 'plane', p].edge_index, _ = dropout_edge(data[p, 'plane', p].edge_index, p=0.05, force_undirected=True)

            ## Test 4: Reducing node degree to a maximum by randomly pruning edges
            elif rewire_type == 'maxdeg':
                data[p, 'plane', p].edge_index = Rewiring.limit_node_degree(data, num_nodes=n_nodes, plane=p,
                                                                            max_degree=12, reconnect=False)

            ## Test 5: Reducing node degree and adding random edges to conserve total number of edges
            elif rewire_type == 'maxdeg_reconnect':
                data[p, 'plane', p].edge_index = Rewiring.limit_node_degree(data, num_nodes=n_nodes, plane=p,
                                                                            max_degree=12, reconnect=True)

            ## Test 6: Reducing node degree by pruning connections with nodes at intermediate distances
            elif rewire_type == 'maxdeg_meddist':
                data[p, 'plane', p].edge_index = Rewiring.limit_node_degree(data, num_nodes=n_nodes, plane=p,
                                                                            max_degree=12, reconnect=False, prune='median')

            ## Test 7: Reducing node degree by uniformly pruning nodes according to distances but keeping extreme ones
            elif rewire_type == 'maxdeg_unif':
                data[p, 'plane', p].edge_index = Rewiring.limit_node_degree(data, num_nodes=n_nodes, plane=p,
                                                                            max_degree=10, reconnect=False, prune='uniform')

            else:
                raise Exception('invalid rewiring type')

        return data

    @staticmethod
    def to_directed(edges: torch.Tensor) -> torch.Tensor:
        """Returns the directed version of an undirected edge tensor. It can also be used to elimite duplicate edges.

           OBS: If the undirected pair is next to each other in the edge index tensor this function can be made much faster
                since we can drop the unique and simply do torch.sort(edges, dim=0)[0][:,::2]. This function is currently
                the major bootleneck in the permute_edges function and it is applied twice.
        """
        return coalesce(edges.sort(dim=0)[0]) # torch.unique(torch.sort(edges, dim=0)[0], dim=1) is 10x slower

    @staticmethod
    def reconnect_isolated_nodes(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Reintegrate isolated nodes into the graph

           OBS: In the current implementation it only works if the number of isolated nodes is less than or equal to the
                number of non-isolated nodes. This assumption was made to speed up the function because it is always
                true in NuGraph's case.
        """
        new_edges = edges.detach().clone()

        if contains_isolated_nodes(edges, num_nodes=num_nodes):
            # Since nodes are isolated I have the freedom to conenct it to any other node besides itself without worrying
            # about duplicated edges
            possible_edges = torch.arange(num_nodes)
            mask_connected_nodes = torch.isin(possible_edges, edges)

            # Find the id's of the isolated nodes
            id_isolated_nodes = possible_edges[~mask_connected_nodes]

            # Connect the isolated to the remaining nodes to eliminate both the possibility of self-connections, and
            # isolated nodes connecting to each other forming disconnected subgraphs
            id_nodes_to_connect = possible_edges[mask_connected_nodes]
            id_nodes_to_connect = id_nodes_to_connect[torch.randperm(id_nodes_to_connect.size(0))][:id_isolated_nodes.size(0)]

            new_connections = torch.stack((id_isolated_nodes, id_nodes_to_connect), dim=0)
            new_connections, _ = torch.sort(new_connections, dim=0)

            new_edges = torch.cat((new_edges, new_connections), dim=1)

        return new_edges

    @staticmethod
    def permute_edges(edges: torch.Tensor, num_nodes: int, algo: int=1) -> torch.Tensor:
        """Permute edges and return a new edge index with the same size as the original one

           OBS1: This function assumes that the incoming edge tensor is undirected

        """
        num_unique_edges = int(edges.size(1) / 2)

        if algo == 1:
            new_edges = Rewiring.to_directed(edges)
            new_edges[0] = new_edges[0, torch.randperm(num_unique_edges)]
        elif algo == 2:
            new_edges = torch.randint(0, num_nodes, size=(2, num_unique_edges))

        # Remove self loops and turn the graph into directed again since undirected pairs can appear with the
        # permutation. The directed function also eliminate possible duplicate edge pairs.
        new_edges = Rewiring.to_directed(remove_self_loops(new_edges)[0])

        # Reconnect isolated nodes to the graph
        new_edges = Rewiring.reconnect_isolated_nodes(new_edges, num_nodes)

        # Iterate until the new edge tensor is of the same size as the original edge tensor
        while new_edges.size(1) < num_unique_edges:
            # We could try to generated all remaining edge candidates in one take, but it could get stuck if the number
            # of edges to add is large. I don't expect num_to_fill to be a very large number, so this loop won't take long.
            # num_edges_to_add = num_unique_edges - new_edges.size(1),
            ij = torch.randint(0, num_nodes, size=(2, 1))
            if ij[0] != ij[1]:
                # This works because there are no self-connections (i,i)
                ij_in_edge = torch.isin(new_edges, ij).all(dim=0)

                if not ij_in_edge.any():
                    new_edges = torch.cat((new_edges, ij), dim=1)

        # Turn the edge indices tensor back into undirected
        new_edges = to_undirected(new_edges, num_nodes=num_nodes)

        return new_edges

    @staticmethod
    def limit_node_degree(data: "pyg.data.HeteroData", num_nodes: int, max_degree: int, plane: str, reconnect: bool = False,
                          prune: str = 'uniform') -> torch.Tensor:
        """Takes an undirected edge and randomly prune edges of nodes with a degree greater than `max_degree`. If
           reconnect is true, redistribute the pruned edges across the network and conserve the number of edges by
           adding random connections.

          OBS1: Assuming the edge tensor indexes are [0, num_nodes -1] and that there are no isolated nodes in
                the graph
        """
        edges = data[plane, 'plane', plane].edge_index

        # Turns undirected edge index into directed
        new_edges = Rewiring.to_directed(edges)
        num_unique_edges = int(edges.size(1) / 2)

        # It is easier to find the node degree using the undirected edges
        nodes_degree = torch.unique(edges[0], return_counts=True, sorted=True)[1]
        # There are no isolated nodes so nodes_degree is always of the same size and has no zero entries

        # Finding the indexes of the nodes with degree greater and smaller than max_degree
        degree_mask = nodes_degree > max_degree
        id_node_to_prune = torch.nonzero(degree_mask) # This works because nodes_degree is sorted according to node index
        id_node_to_connect = torch.nonzero(~degree_mask)

        # Prune the edges of nodes whose degree is greater than the established value
        if prune == 'random':
            for id_node in id_node_to_prune:
                edges_mask = torch.isin(new_edges, id_node).any(dim=0)
                idxs_to_keep = torch.nonzero(edges_mask).view(-1) # Indexes of the columns that has 'id_node'
                idxs_to_keep = idxs_to_keep[torch.randperm(idxs_to_keep.size(0))][:max_degree] # keeping 'max_degree' edges
                edges_mask[idxs_to_keep] = False
                new_edges = new_edges[:, ~edges_mask] # Remove the randomly chosen edge columns
            # We don't need to check idxs_to_keep > max_degree because even if it is smaller, [:maxdegree] will just
            # return the entire array

        elif prune == 'median' or prune == 'uniform':
            # Extracting x=wire and y=time information from the graph
            wt_coords = torch.stack((data.collect("pos")[plane][:, 0], data.collect("pos")[plane][:, 1]), dim=1)

            # Calculating pairwise euclidean distances of nodes in the wire vs time space
            dist_table = torch.norm(wt_coords[:, None, :] - wt_coords[None, :, :], dim=-1)
            dist_table.fill_diagonal_(float('inf'))

            for id_node in id_node_to_prune:
                # Finding the edges that contain 'id_node'
                edges_mask = torch.isin(new_edges, id_node).any(dim=0)
                idxs_to_keep = torch.nonzero(edges_mask).view(-1) # indexes of the edges of 'id_node' w.r.t. 'new_edges'
                edges_id_node = new_edges[:, idxs_to_keep]

                # We check this because the node degree of a node can decrease due to pruning of previous nodes and
                # become smaller than 'max_degree'. We cannot update id_node_to_prune on real time because we are
                # iterating over it. The solution is to either use this 'if' or use unique(idxs_to_keep) but the
                # latter is slow
                if idxs_to_keep.size(0) > max_degree:
                    # Finding the indexes of the nodes connected to 'id_node' sorted by distance in wire vs time space
                    dists = dist_table[edges_id_node[0], edges_id_node[1]]
                    _, distsorted_idxs = torch.sort(dists)

                    # Sorting the idxs_to_keep by node distance in wire vs time space
                    idxs_to_keep = idxs_to_keep[distsorted_idxs]
                    # Note we don't need to sort the edges as edges_id_node[:,distsorted_idxs], just idxs_to_keep

                    if prune == 'median':
                        # Removing intermediate distances from 'idxs_to_keep' and keeping only 'max_degree' indexes
                        idxs_to_keep = torch.cat((idxs_to_keep[:ceil(max_degree/2)],
                                                  idxs_to_keep[-max_degree//2:]))

                    elif prune == 'uniform':
                        # Uniformly sampling idxs_to_keep but keeping the extreme edges, and then randomly pruning extra
                        # edges to reach max_degree
                        step = idxs_to_keep.size(0) // max_degree
                        idxs_to_keep = torch.cat((idxs_to_keep[0].unsqueeze(0),
                                                  idxs_to_keep[1:-1:step],
                                                  idxs_to_keep[-1].unsqueeze(0)), dim=0)
                        if idxs_to_keep.size(0) > max_degree:
                            # Excluding first and last indexes, corresponding to the shortest and longest edges
                            aux_idxs = torch.arange(1, idxs_to_keep.size(0) - 1)
                            aux_idxs = aux_idxs[torch.randperm(aux_idxs.size(0))][:max_degree-2]
                            aux_idxs = torch.cat((torch.tensor([0, idxs_to_keep.size(0)-1]), aux_idxs), dim=0)
                            idxs_to_keep = idxs_to_keep[aux_idxs]

                    # Pruning the edge pairs
                    edges_mask[idxs_to_keep] = False
                    new_edges = new_edges[:, ~edges_mask]

        # Reconnect any nodes that were isolated during the pruning
        new_edges = Rewiring.reconnect_isolated_nodes(new_edges, num_nodes)
        # We don't check for the node degree after rewiring, so in theory we may end up with nodes with degree greater
        # than `max_degree`. However, in NuGraph this won't be a big problem since the chance to have isolated nodes is
        # small and given there are isolated nodes the chance to connect to a node whose degree is `max_degree` is very
        # tiny (there are on average 1500 nodes per event).

        # Add random connections to ensure that the returned edge index is of the same size as the input edge index
        if reconnect == 'random':
            # Iterate until the new edge tensor is of the same size as the original edge tensor
            while new_edges.size(1) < num_unique_edges:
                rand_idx = torch.randint(0, id_node_to_connect.size(0), size=(2,))
                ij = torch.tensor([ [ id_node_to_connect[rand_idx[0]] ],
                                    [ id_node_to_connect[rand_idx[1]] ] ])

                if ij[0] != ij[1] and nodes_degree[ij[0]] < max_degree and nodes_degree[ij[1]] < max_degree:
                    # This works because there are no self-connections (i,i)
                    ij_in_edge = torch.isin(new_edges, ij).all(dim=0)

                    if not ij_in_edge.any():
                        new_edges = torch.cat((new_edges, ij), dim=1)
                        nodes_degree[ij[0]] += 1
                        nodes_degree[ij[1]] += 1

        # Turn the edge indices tensor back into undirected
        new_edges = to_undirected(new_edges)

        # new_nodes_degree = torch.unique(new_edges[0], return_counts=True, sorted=True)[1]
        # print(nodes_degree.sum(), new_nodes_degree.sum())

        return new_edges
