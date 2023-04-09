# MEW Experiment Design

> `rsm_pmax=0.1` as a control variable (based on experiments/explore_scalability/analyze.ipynb)

### Posets

- m = 10
- n = [100, 1000, 10000]
- rule = [plurality, 2-approval, Borda]
- ~~Fix phi = 0.5, change pmax = [0.1, 0.5, 0.9]~~
- Fix pmax = 0.1, change phi = [0.1, 0.5, 0.9]
- Voter pruning at profile level
- For each voter, run the (RIM, poset) solver in Section 7.1.

Additional sub-figure for cover width

- m = 10, phi = 0.5, change pmax = [0.1, 0.5, 0.9]
- Generate 1000 partial orders
- Draw a figure for runtime vs cover_width

### Partitioned Preferences with Missing Items

- num_buckets = [5, 10, 20], m=80, n=1000, rule = [plurality, 2-approval, Borda]
- num_buckets = 5, m = [10, 20, 40, 80], n = 1000, rule = [plurality, 2-approval, Borda]
- num_buckets = 5, n = [100, 1000, 10000], m = 10, rule = [plurality, 2-approval, Borda]
- Voter pruning at profile level
- In the figure, y = runtime across the entire profile

### Partitioned Preferences

- num_buckets = [5, 10, 20], m=80, n=1000, rule = [plurality, 2-approval, Borda]
- num_buckets = 5, m = [10, 20, 40, 80], n = 1000, rule = [plurality, 2-approval, Borda]
- num_buckets = 5, n = [100, 1000, 10000], m = 10, rule = [plurality, 2-approval, Borda]
- Voter pruning at profile level

### Partial chains

- chain_size = [5, 10, 20], m=80, n=1000, rule = [plurality, 2-approval, Borda]
- chain_size = 5, m = [10, 20, 40, 80], n = 1000, rule = [plurality, 2-approval, Borda]
- chain_size = 5, n = [100, 1000, 10000], m = 10, rule = [plurality, 2-approval, Borda]
- Voter pruning at profile level

### Truncated rankings

- m = 80, top_size = bottom_size = [5, 10, 20], n = 1000, rule = [plurality, 2-approval, Borda]
- m = 80, top_size = bottom_size = 5, n = [100, 1000, 10000], rule = [plurality, 2-approval, Borda]
- Voter pruning at profile level

## Probabilistic profiles

### Mallows

- m = [10, 20, 40, 80], phi=0.5, n = 1000, rule = [plurality, 2-approval, Borda]
- m = 10, phi=[0.1, 0.5, 0.9], n = 1000, rule = [plurality, 2-approval, Borda]
- m = 10, phi=0.5, n = [100, 1000, 10000], rule = [plurality, 2-approval, Borda]
- Voter pruning at profile level

### rRSM

- m = [10, 20, 40, 80], pij_matrix = mallows.phi = 0.5, n = 1000, rule = [plurality, 2-approval, Borda]
- m = 10, pij_matrix = mallows.phi = [0.1, 0.5, 0.9], n = 1000, rule = [plurality, 2-approval, Borda]
- m = 10, pij_matrix = mallows.phi = 0.5, n = [100, 1000, 10000], rule = [plurality, 2-approval, Borda]
- Voter pruning at profile level

## Combined profiles

### Mallows + Partitioned Preferences

- m = 10, phi=[0.1, 0.5, 0.9], num_buckets = 2, n=1000, rule = [plurality, 2-approval, Borda]

### Mallows + Truncated Rankings

- m = 10, phi=[0.1, 0.5, 0.9], top = bottom = 3, n=1000, rule = [plurality, 2-approval, Borda]

### Mallows + Posets

- Fix center, m = 10, phi=[0.1, 0.5, 0.9], posets generated by pmax = 0.9, rule = [plurality, 2-approval, Borda]

## Real dataset

- MovieLens
  
     - Partitioned preferences with missing items

- CrowdRank
  
     - For each HIT (20 items), use solver for partial chain

- Travel
  
     - [UCI Machine Learning Repository: Tarvel Review Ratings Data Set](https://archive.ics.uci.edu/ml/datasets/Tarvel+Review+Ratings)
  
     - Dessert (break cycles, like the NW PW paper)

## TODO

- Run the experiments, write up the dataset description, add experiment figures.
- Add the section of voter pruning

---

## Figures

Speedup vs #voters, with #candidates fixed

- #voters = [10, 100, 1_000, 10_000, 100_000]
- Generate 5 profiles for each #voters, and #candidates=10.

Speedup vs #candidates, with #voters = 1_000

- #candidates = [5, 6, 7, 8, 9, 10]
- Generate 5 profiles for each #candidates, and #voters=1_000

## Greene Specs

- Standard Memory Compute Nodes (cs)
     - Node type: Lenovo SD650 Water-cooled nodes on two-node trays
     - CPU: **48 processing cores** per node: **2x** Intel Xeon Platinum 8268 **24C** 205W 2.9GHz Processor
     - Memory: Total: **192GB** - 12x16GB DDR4, 2933MHz; Available to user jobs: **180GB**
     - Local disk: 1x 480GB M.2 SSD
     - Infiniband interconnect: Shared IO Mellanox ConnectX-6 HDR IB/200Gb VPI 1-Port x16 PCIe 4.0 HCA
- Medium Memory Compute Nodes (cm)
     - Node type: Lenovo SD650 Water-cooled nodes on two-node trays
     - CPU: **48 processing cores** per node: **2x** Intel Xeon Platinum 8268 **24C** 205W 2.9GHz Processor
     - Memory: Total: **384GB** - 12x 32GB DDR4, 2933MHz; Available to user jobs: **369GB**
     - Local disk: 1x 480GB M.2 SSD
     - Infiniband interconnect: Shared IO Mellanox ConnectX-6 HDR IB/200Gb VPI 1-Port x16 PCIe 4.0 HCA
- Large Memory Compute Nodes (cl)
     - Node type: Lenovo SR850
     - CPU: 96 processing cores per node: 4 socket Intel Xeon Platinum 8268 24C 205W 2.9GHz Processor
     - Memory: Total: 3,092GB - 48x 64GB DDR4, 2933MHz; Available to user jobs: 3014GB
     - Local disk: 1x 1.92TB SSD
     - Infiniband interconnect: 1x Mellanox ConnectX-6 HDR100 /100GbE VPI 1-Port x16 PCIe 3.0 HCA
- Benny's machine
     - Intel® Xeon® Processor E5-2680 v3 (30M Cache, 2.50 GHz)

# Appendix



### Datasets in previous papers

Kenig, etal. Probabilistic Inference Over Repeated Insertion Models. AAAI 2018.

- Synthetic data
     - 3 reference rankings where $m = [30, 60, 100]$. There are 250 partial orders generated for each $m$.
     - Let $p_V$ and $p_E$ denote probabilities that an item and an edge is added to the poset. First select items by $p_V$, then edges are retained by $p_E$.
          - $m=30, p_V \in [0.3, 0.9], p_E \in [0.05, 0.3] \rightarrow 3 \leq |\nu| \leq 29$
          - $m=60, p_V \in [0.1, 0.5], p_E \in [0.1, 0.8] \rightarrow 2 \leq |\nu| \leq 16$
          - $m=100, p_V \in [0.06, 0.09], p_E \in [0.1, 0.8] \rightarrow 2 \leq |\nu| \leq 13$

Noothigattu, etal. A Voting-Based System for Ethical Decision Making. AAAI 2018.

- Synthetic Data
- Moral Machine Data (not public)
     - ~1.3 million voters
     - ~18 million pairwise comparisons

Zhao, etal. A Cost-Effective Framework for Preference Elicitation and Aggregation. arXiv:1805.05287, 2018.

- Amazon Mechanical Turk (MTurk) collected data (not public)
     - full rankings (2 <= length <= 10)
     - top-$k$ lists (k=[1, 10] for 10 hotels)
- Synthetic Data

Chakraborty, etal. Algorithmic Techniques for Necessary and Possible Winners. arXiv:2005.06779, 2020.

- Google Travel Ratings
     - 5456 users
     - 24 travel categories
     - Average ratings given by users for each travel category (averaged across the travel sites within that category)
- Dessert dataset
     - 228 users
     - 8 desserts
     - pairwise comparisons

### Preference types

| Task     | Parameters            | Complexity             | Eva         |
| -------- | --------------------- | ---------------------- | ----------- |
| PP       | m, #buckets           | O(1) per item per rank | Trivial     |
| PC       | m, #items             | O(1) per item per rank | Trivial     |
| RIM      | m, randomized \Pi     | O(m^3) per item        | Interesting |
| rRSM     | m, randomized \Pi     |                        | Interesting |
| RIM + TR | m, top t, bottom b    |                        | Fine        |
| MAL + PP | m, \phi, PP benchmark |                        | Interesting |

### Strategy - candidate pruning

1. Calculate upper and lower score bounds for each candidate over each voter

2. From voter 1 to voter n,
   
      1. Remove candidates who will surely lose;
      2. Calculate exact scores for each candidate.

### Strategy of voter clustering

1. Input
   
      1. `posets: list[Poset]`
   
      2. `rule_vector: list[int]`
   
      3. `d`, voter clustering by top-`d` candidates

2. Quick voter clustering
   
      1. Quickly compute extremal ranks of each candidate over each voter
   
      2. Assume a uniform distribution between extremal ranks.
   
      3. Combine with `rule_vector` to obtain candidate score estimation. (tie-breaking by candidate names)
   
      4. Cluster by voters by top-`d` candidates, and obtain `cluster_to_voters: dict[tuple, list[Poset]]`

3. Maintain
   
      1. `cluster_to_unfinished_candidates: dict[tuple, set]`
   
      2. `cluster_to_candidate_to_exact_score: dict[tuple, dict[int, float]]`
   
      3. `cluster_to_candidate_to_upper_score: dict[tuple, dict[int, float]]`
   
      4. `cluster_to_candidate_to_lower_score: dict[tuple, dict[int, float]]`
   
      5. `candidate_to_upper_score: dict[int, float]`
   
      6. `candidate_to_lower_score: dict[int, float]`

4. Prune candidates by accumulated upper / lower bounds

5. While True:
   
      1. For each cluster, if some unknown candidate remains, do candidate pruning (sort them by their upper and lower bounds)
      2. Prune candidates by accumulated upper / lower bounds
      3. if `cluster_to_unfinished_candidates` all empty: break loop

6. Return winners

### A hard case for MIS-AMP-adaptive

```python
# A hard case
Poset(parent_to_children={83: {115}, 103: {98}, 98: {3}, 102: {104}, 51: {19}, 65: {60}, 11: {69}}, item_set={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119})
rank_probs_of_item_3 = [0.0, 0.0, 3.5607463324309572e-06, 1.0682238997294078e-05, 2.1364477994588356e-05, 3.560746332431711e-05, 5.341119498647442e-05, 7.477567298105843e-05, 9.970089730808192e-05, 0.00012818686796754012, 0.00016023358495939553, 0.00019584104828371355, 0.00023500925794047093, 0.00027773821392967674, 0.0003240279162512783, 0.0003738783649053032, 0.00042728955989178655, 0.00048426150121066915, 0.0005447941888620342, 0.0006088876228458188, 0.0006765418031619292, 0.0007477567298105219, 0.0008225324027916083, 0.0009008688221051395, 0.0009827659877509913, 0.0010682238997293525, 0.0011572425580402236, 0.0012498219626834554, 0.0013459621136590143, 0.0014456630109670585, 0.001548924654607614, 0.001655747044580612, 0.0017661301808860254, 0.0018800740635238212, 0.001997578692494068, 0.0021186440677968144, 0.0022432701894319683, 0.0023714570573993825, 0.0025032046716992767, 0.0026385130323316806, 0.002777382139296411, 0.0029198119925935248, 0.0030658025922231766, 0.003215353938185394, 0.0033684660304801385, 0.003525138869107289, 0.003685372454066778, 0.0038491667853585948, 0.004016521862982822, 0.004187437686939598, 0.004361914257228784, 0.004539951573850302, 0.004721549636804401, 0.0049067084460910695, 0.005095428001709935, 0.005287708303660899, 0.005483549351944291, 0.0056829511465604065, 0.005885913687509141, 0.006092436974790209, 0.00630252100840362, 0.006516165788349633, 0.006733371314628208, 0.006954137587239063, 0.007178464606182285, 0.007406352371458077, 0.007637800883066182, 0.00787281014100646, 0.008111380145279297, 0.008353510895884837, 0.008599202392822611, 0.008848454636092426, 0.009101267625694817, 0.009357641361630174, 0.009617575843898066, 0.009881071072497723, 0.010148127047429092, 0.010418743768693178, 0.010692921236290679, 0.010970659450220839, 0.011251958410482866, 0.011536818117077564, 0.011825238570005054, 0.012117219769264198, 0.012412761714855771, 0.01271186440678037, 0.01301452784503685, 0.013320752029625585, 0.013630536960547257, 0.013943882637801297, 0.014260789061387788, 0.014581256231306777, 0.014905284147558285, 0.015232872810141776, 0.015564022219057115, 0.01589873237430552, 0.016237003275887327, 0.016578834923800204, 0.01692422731804634, 0.017273180458623444, 0.017625694345534435, 0.01798176897877937, 0.0183414043583536, 0.01870460048426017, 0.01907135735650042, 0.019441674975073136, 0.019815553339976956, 0.02019299245121483, 0.02057399230878481, 0.020958552912688246, 0.02134667426292383, 0.021738356359490105, 0.022133599202389716, 0.022532402791624242, 0.022934767127187207, 0.023340692209086647, 0.02375017803731715, 0.0241632246118764, 0.024579831932772165, 0.025000000000002263]
```