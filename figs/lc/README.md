## Algorithms' learning curves with final rankings

In this folder, we provide insights into how ranking of algorithms change w.r.t. the training data size (see an example in Figure 1). Algorithms are color-coded based on their final ranking. It can be seen that a group of algorithms represented in dark blue dominates throughout the learning curve.  In other words, algorithm that ranked first initially tends to maintain a very high rank by the end of the process across most datasets. In such cases, BOS baseline can be more efficient compared to DDQN because it can achieve comparable/better performances while being easier to implement.


![Fig 1](flora.svg)

*Fig 1: **Algorithm learning curves with final ranking on dataset *flora***. They are colored by their final rankings (last points on the curves). Algorithms highly ranked at the beginning (dark blue) often finished with high ranks as well.