# Graph-MPNN-transformer
## MPNN that uses the transformer encoder as neighbourhood aggregator function.

Starting from <a href=https://arxiv.org/abs/1706.02216>graphSAGE</a> <a href=https://github.com/williamleif/graphsage-simple>simple implementation</a> a transformer encoder is used as trainable aggregation function rather than a convolution.
Inspired by <a href="https://arxiv.org/pdf/2007.14062.pdf"> BigBird </a> global nodes are generated and added to the network allowing the transformer to attend positions more distant than 1-hop neighborhood.
Louvain partition algorithm hierarchically aggregates communities in a greedy approach maximizing modularity. Roughly speaking it aggregates the densely connected communities hierarchiacally and chosing the granularity level as hyperparameter it provides a partitioning. A global node is generated for each detected community and it is connected to all the nodes that belong to the partition and to the other global nodes.
Despite Louvain is an efficient way to provide a possible partitining other methods can still be evaluated.
Global nodes' features are initialized as the maximum pooling or the mean pooling of all the nodes in the partition they represent.  
Despite the signal provided by global nodes is very coarse, it can still provide some minor improvements. Ideally, we want each node to attend also on nodes farther than one or two hops apart, but this is computationally prohibitive since networks size is several orders of magnitude larger than text sentences.

To further help the network capturing structural property of the graph laplacian positional encoding is introduced as in "<a href=https://arxiv.org/abs/2012.09699>A Generalization of Transformer Networks to Graphs
</a>" by injecting graph structural information into the nodes features. 
## Hyperparameters
```
global_nodes:
    - description: Add global nodes node generated from communities detection
    - choices: [True, False]
global_nodes_connections:
    - description: Connect globals node with either the nodes in the partition and other global nodes or connect global nodes with all nodes in the network
    - choices : ['all', 'partitions']
laplacian_encoding:
    - description: Use laplacian encoding to embed node position in the network in node features
    - choices : [True, False]
emb_size :
    - description: Embedding size of input linear projection to deal with one hot encoding high dimensionality
n_head :
    - description: Number of attention heads
dropout :
    - description: Dropout probability rate
num_samples :
    - description: Number of neighbours to sample in doing node aggregations. Use null to take all neighbors.
layers :
    - description: Network depth
split : 'gcn'
    - description: Train, val and test split. The available options are the split used in the gcn paper(https://arxiv.org/abs/1609.02907) or the split used in Caleynet paper (a href=https://arxiv.org/abs/1705.07664)
    - choices : ['gcn', 'caley']
optimizer :
    - description: Optimizer used
    - choices : ['adam', 'sgd']
sgd_momentum :
    - description: momentum of sgd optimizer
weight_decay :
    - description: weight decay
lr :
    - description: learning rate
```
