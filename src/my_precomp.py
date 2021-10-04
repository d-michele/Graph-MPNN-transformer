from typing import List

import numpy as np
from numba import get_num_threads, prange
from numba.np.ufunc import _get_thread_id
from pecanpy.node2vec import PreComp


class MyPreComp(PreComp):
    def simulate_walks(self, nodes: List, num_walks: int, walk_length: int):
        """Generate walks starting from each nodes ``num_walks`` time.

        Note:
            This is the master process that spawns worker processes, where the
            worker function ``node2vec_walks`` genearte a single random walk
            starting from a vertex of the graph.

        Args:
            nodes : nodes from which we start the random walk
            num_walks : number of walks starting from each node.
            walks_length : length of walk.

        """
        num_nodes = len(self.IDlst)
        start_node_idx_ary = np.concatenate([nodes] * num_walks)
        np.random.shuffle(start_node_idx_ary)

        move_forward = self.get_move_forward()
        has_nbrs = self.get_has_nbrs()
        verbose = self.verbose

        @jit(parallel=True, nogil=True, nopython=True)
        def node2vec_walks():
            """Simulate a random walk starting from start node."""
            n = start_node_idx_ary.size
            # use last entry of each walk index array to keep track of effective walk length
            walk_idx_mat = np.zeros((n, walk_length + 2), dtype=np.uint32)
            walk_idx_mat[:, 0] = start_node_idx_ary  # initialize seeds
            # set to full walk length by default
            walk_idx_mat[:, -1] = walk_length + 1

            # progress bar parameters
            n_checkpoints = 10
            checkpoint = n / get_num_threads() // n_checkpoints
            progress_bar_length = 25
            private_count = 0

            for i in prange(n):
                # initialize first step as normal random walk
                start_node_idx = walk_idx_mat[i, 0]
                if has_nbrs(start_node_idx):
                    walk_idx_mat[i, 1] = move_forward(start_node_idx)
                else:
                    walk_idx_mat[i, -1] = 1
                    continue

                # start bias random walk
                for j in range(2, walk_length + 1):
                    cur_idx = walk_idx_mat[i, j - 1]
                    if has_nbrs(cur_idx):
                        prev_idx = walk_idx_mat[i, j - 2]
                        walk_idx_mat[i, j] = move_forward(cur_idx, prev_idx)
                    else:
                        walk_idx_mat[i, -1] = j
                        break

                if verbose:
                    # TODO: make monitoring less messy
                    private_count += 1
                    if private_count % checkpoint == 0:
                        progress = private_count / n * progress_bar_length * get_num_threads()

                        # manuual construct progress bar since string formatting not supported
                        progress_bar = '|'
                        for k in range(progress_bar_length):
                            progress_bar += '#' if k < progress else ' '
                        progress_bar += '|'

                        print("Thread # " if _get_thread_id() < 10 else "Thread #",
                              _get_thread_id(), "progress:", progress_bar,
                              get_num_threads() * private_count * 10000 // n / 100, "%")

            return walk_idx_mat

        walks = [[self.IDlst[idx] for idx in walk[:walk[-1]]]
                 for walk in node2vec_walks()]

        return walks
