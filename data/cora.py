from collections import defaultdict

import dgl.data

from torch_geometric.datasets import Planetoid
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree

import time

import dgl
import torch

from scipy import sparse as sp
import numpy as np
import networkx as nx

def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in SBMsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges())

    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass

    return full_g

def to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=None):
    r""" from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
    """
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = maybe_num_nodes(edge_index, num_nodes)
    out = sp.coo_matrix(
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out

def laplacian_positional_encoding(g, pos_enc_dim, library):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    if library == "dgl":
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    else:
        A = to_scipy_sparse_matrix(g.edge_index).astype(float)
        N = sp.diags(dgl.backend.asnumpy(degree(g.edge_index[0]).int()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.num_nodes) - N * A * N

        # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        g['ndata']['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return g

class CoraDataset(torch.utils.data.Dataset):

    def __init__(self, name, library=dgl):
        """
            Loading Cora datasets
        """
        start = time.time()
        self.library = library
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        if library == 'dgl':
            self.data = dgl.data.CoraGraphDataset()
            self.graph = self.data[0]
            print('num nodes / edges :', self.graph.num_nodes(), self.graph.num_edges())
        else: #pytorch geometric (p_geo)
            self.data = Planetoid(root='/tmp/Cora', name='Cora')
            self.graph = self.data.data
            self.graph['ndata'] = defaultdict(dict)
            print('num nodes / edges :', self.graph.num_nodes, self.graph.num_edges)

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph = laplacian_positional_encoding(self.graph, pos_enc_dim, self.library)





