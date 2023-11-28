# MUSE
Code and datasets for paper **MUSE: Multi-Knowledge Passing on the Edges, Boosting Knowledge Graph Completion**

To run on different datasets, you can modify the parameters in main.py

```
python main.py
```
Some results of MUSE in relation prediction in general scenario:
|Datasets    | mrr              | h@1              | h@3              |
|-----------|------------------|------------------|-------------------|
| FB15k-237| 0.9853 ± 0.0003  | 0.9741 ± 0.0006  | 0.9968 ± 0.0002  |
| wn18    | 0.9949 ± 0.0007  | 0.9911 ± 0.0011  | 0.9987 ± 0.0003  |
| wn18rr   | 0.9868 ± 0.0011  | 0.9743 ± 0.0022  | 0.9999 ± 0.0001  |
| NELL995 | 0.9393 ± 0.0016  | 0.8975 ± 0.0025  | 0.9794 ± 0.0019  |

Relation Prediction of MUSE in the Limited Information Set (LIS) and Rich Information Set (RIS) Scenarios:
Datasets    |   PathCon-LIS  | PathCon-RIS    |
|-----------|-------------------|-------------------|
| FB15k-237| 0.9751 ± 0.0066   | 0.9772 ± 0.0006   |
| wn18    | 0.9955 ± 0.0014   | 0.9906 ± 0.0012   |
| wn18rr   | 0.9609 ± 0.0045   | 0.9784 ± 0.0018   |
| NELL995  | 0.8585 ± 0.0075   | 0.9270 ± 0.0012   |
