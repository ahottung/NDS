from src import InstanceGenCVRP, InstanceGenVRPTW, InstanceGenPCVRP

import torch
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default='cvrp', help="cvrp or vrptw or pcvrp")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset")
    parser.add_argument('--n', type=int, default=100, help="Sizes of problem instances")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")

    # CVRP
    parser.add_argument('--depot_pos', type=int, default=2)
    parser.add_argument('--cus_pos', type=int, default=2)
    parser.add_argument('--demand_dis', type=int, default=2)
    parser.add_argument('--avg_route_size', type=int, default=3)

    # Additional TW parameters
    parser.add_argument('--service_window', type=int, default=2400)
    parser.add_argument('--time_window_size', type=int, default=500)
    parser.add_argument('--service_duration', type=int, default=50)

    # Additional PC parameters
    parser.add_argument('--prize_min', type=float, default=0.01)
    parser.add_argument('--prize_max', type=float, default=0.1)
    parser.add_argument('--prize_alpha', type=float, default=0.5)


    opts = parser.parse_args()

    os.makedirs(opts.data_dir, exist_ok=True)

    if opts.problem == 'cvrp':
        filename = os.path.join(opts.data_dir, "cvrp{}_{}_seed{}_{}_{}_{}_{}.pt".format(opts.n, opts.name,
                                                                                         opts.seed, opts.depot_pos,
                                                                                         opts.cus_pos, opts.demand_dis,
                                                                                         opts.avg_route_size))

        instanceGen = InstanceGenCVRP(opts.n, True, opts.depot_pos, opts.cus_pos, opts.demand_dis,
                                      opts.avg_route_size)

        depot_xy, node_xy, node_demand, capacity = instanceGen.get_random_problems(opts.dataset_size, opts.seed)


        torch.save({
            'depot_xy': depot_xy,
            'node_xy': node_xy,
            'node_demand': node_demand,
            'capacity': capacity,
            'grid_size': 1
        }, filename)

        print(f"Saved CVRP dataset to {filename}")

    elif opts.problem == 'vrptw':
        filename = os.path.join(opts.data_dir, "vrptw{}_{}_seed{}_{}_{}_{}_{}_{}_{}_{}.pt".format(opts.n, opts.name,
                                                                                               opts.seed,
                                                                                               opts.depot_pos,
                                                                                               opts.cus_pos,
                                                                                               opts.demand_dis,
                                                                                               opts.avg_route_size,
                                                                                               opts.service_window,
                                                                                               opts.time_window_size,
                                                                                               opts.service_duration))

        instanceGen = InstanceGenVRPTW(opts.n, True, opts.depot_pos, opts.cus_pos, opts.demand_dis,
                                       opts.avg_route_size, opts.service_window, opts.time_window_size,
                                        opts.service_duration)

        depot_xy, node_xy, node_demand, capacity, depot_tw, node_tw, node_sd = instanceGen.get_random_problems(opts.dataset_size, opts.seed)

        torch.save({
            'depot_xy': depot_xy,
            'node_xy': node_xy,
            'node_demand': node_demand,
            'capacity': capacity,
            'depot_tw': depot_tw,
            'node_tw': node_tw,
            'node_sd': node_sd,
            'grid_size': 1
        }, filename)

        print(f"Saved VRPTW dataset to {filename}")

    elif opts.problem == 'pcvrp':
        filename = os.path.join(opts.data_dir, "pcvrp{}_{}_seed{}_{}_{}_{}_{}_{}_{}_{}.pt".format(opts.n, opts.name,
                                                                                               opts.seed,
                                                                                               opts.depot_pos,
                                                                                               opts.cus_pos,
                                                                                               opts.demand_dis,
                                                                                               opts.avg_route_size,
                                                                                               opts.prize_min,
                                                                                               opts.prize_max,
                                                                                               opts.prize_alpha))

        instanceGen = InstanceGenPCVRP(opts.n, True, opts.depot_pos, opts.cus_pos, opts.demand_dis,
                                       opts.avg_route_size, opts.prize_min, opts.prize_max, opts.prize_alpha)

        depot_xy, node_xy, node_demand, capacity, node_prize = instanceGen.get_random_problems(opts.dataset_size, opts.seed)

        torch.save({
            'depot_xy': depot_xy,
            'node_xy': node_xy,
            'node_demand': node_demand,
            'capacity': capacity,
            'node_prizes': node_prize,
            'grid_size': 1
        }, filename)

        print(f"Saved PCVRP dataset to {filename}")

    else:
        raise ValueError(f"Problem {opts.problem} not supported")


