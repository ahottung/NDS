# Neural Destruction Search (NDS)

**Official implementation for the paper:** [Neural Deconstruction Search for Vehicle Routing Problems](https://openreview.net/forum?id=bCmEP1Ltwq)

## Overview

Neural destruction search (NDS) combines deep reinforcement learning with traditional optimization techniques to solve various vehicle routing problems (VRPs). It is a large neighborhood search method in which solutions are iteratively improved through deconstruction and reconstruction operations. Solutions are deconstructed via a powerful learned policy trained through reinforcement learning, and reconstructed via a simple greedy algorithm. On our test instances, NDS outperforms even state-of-the-art approaches like HGS and SISRs.

## Supported Problem Variants

1. **CVRP (Capacitated Vehicle Routing Problem)**: Classical VRP with vehicle capacity constraints
2. **VRPTW (Vehicle Routing Problem with Time Windows)**: CVRP extended with customer time window constraints
3. **PCVRP (Prize-Collecting Vehicle Routing Problem)**: VRP where customers have associated prizes and not all need to be visited

## Citation

If you use this code, please cite:

```bibtex
@article{
hottung2025neural,
title={Neural Deconstruction Search for Vehicle Routing Problems},
author={Andr{\'e} Hottung and Paula Wong-Chung and Kevin Tierney},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=bCmEP1Ltwq},
note={}
}
```

## Instances
### Datasets
The `data/` directory contains the validation and test instances used in our experiments:
- **CVRP**: `vrp{size}_{name}_seed{seed}.pkl`
- **VRPTW**: `vrptw{size}_{name}_seed{seed}_{params}.pt`
- **PCVRP**: `pcvrp{size}_{name}_seed{seed}_{params}.pt`

### Instance Generator
New instance sets can be generated via the instance generator:
```bash
python generate_dataset.py
```

## Baselines
Our implementation of the SISRs method will be made available soon in a separate repository.

## Installation

### Prerequisites
- Python 3.8+
- C++ compiler (for building extensions)

### Setup
```bash
# Clone the repository
git clone https://github.com/ahottung/NDS
cd NDS

# Install Python dependencies
pip install torch numpy hydra-core cppimport

# The C++ modules will be compiled automatically on first run
```

## Usage

### Training

Train a neural destruction policy using configuration files, e.g., for CVRP100:

```bash
python train.py cvrp_100.yaml
```

Training configurations are located in `configs/train` with separate files for each problem variant and size:
- `cvrp_100.yaml`, `cvrp_500.yaml`, etc. for CVRP
- `vrptw_100.yaml`, `vrptw_500.yaml`, etc. for VRPTW  
- `pcvrp_100.yaml`, `pcvrp_500.yaml`, etc. for PCVRP

Note that training uses 8 cpu cores by default (can be changed via `env_params.num_processes`).

### Evaluation

Evaluate the provided models on test instances, e.g., for CVRP100:

```bash
python eval.py cvrp_100.yaml
```

Test configurations are located in `configs/eval` with separate files for each problem variant and size.

### Configuration Override

Override configuration parameters from command line:

```bash
python eval.py cvrp_100.yaml tester_params.max_runtime=60
```




