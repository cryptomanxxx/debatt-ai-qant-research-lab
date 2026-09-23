# Experiment 005 — Automatic Topology Search

This is the first search in the lab where candidate topologies are generated
programmatically instead of being hand-picked.

Search space:
- 1 to 3 hidden layers
- allowed widths: 32, 64, 128, 256
- ReLU activation
- maximum parameter budget: 220,000

Every topology satisfying the budget is generated automatically, trained under
the same conditions, evaluated with the Q.ANT CPU backend and added to the
fitness table. The script then computes the accuracy/parameter Pareto front.

Run:

```bash
python experiments/005_auto_topology_search/run.py
```

This run is substantially larger than earlier experiments. Outputs are written
to `local_results/`. CPU timing remains diagnostic only.
