import json
import numpy as np
from tabulate import tabulate
import argparse
from pathlib import Path


def compute_metrics_distance_based(episode_stats, task_success_distance_thresholds, task):
    metrics = {}

    travel_distances = np.array(episode_stats['sum_travel_distance'])
    destination_reached = np.array(episode_stats['sum_destination_reached'])
    time_exceeded = np.array(episode_stats['sum_time_exceeded'])
    is_collision = np.array(episode_stats['sum_is_collision'])
    out_of_lane = np.array(episode_stats['sum_out_of_lane'])
    ttc = np.array(episode_stats['mean_ttc'])
    wpt_dis = np.array(episode_stats['mean_wpt_dis'])
    speed_norm = np.array(episode_stats['mean_speed_norm'])
    lengths = np.array(episode_stats['lengths'])
    scores = np.array(episode_stats['scores'])

    num_episodes = len(travel_distances)
    task_threshold = task_success_distance_thresholds.get(task, 0)

    success_mask = travel_distances >= task_threshold
    success_rate = np.mean(success_mask)

    # Filter out failures after the tasks are successful
    # is_collision[success_mask] = False
    # time_exceeded[success_mask] = False
    #out_of_lane[success_mask] = False

    metrics['task'] = task
    metrics['num_episodes'] = num_episodes
    metrics['success_rate'] = success_rate
    metrics['avg_travel_distance'] = np.mean(travel_distances)
    metrics['avg_destination_reached'] = np.mean(destination_reached)
    metrics['avg_destination_reached_ratio'] = np.sum(destination_reached) / num_episodes
    metrics['avg_time_exceeded'] = np.mean(time_exceeded)
    metrics['avg_time_exceeded_ratio'] = np.sum(time_exceeded) / num_episodes
    metrics['avg_is_collision'] = np.mean(is_collision)
    metrics['avg_is_collision_ratio'] = np.sum(is_collision) / num_episodes
    metrics['avg_out_of_lane'] = np.mean(out_of_lane)
    metrics['avg_out_of_lane_ratio'] = np.sum(out_of_lane) / num_episodes
    metrics['avg_ttc'] = np.mean(ttc)
    metrics['avg_wpt_dis'] = np.mean(wpt_dis)
    metrics['avg_speed_norm'] = np.mean(speed_norm)
    metrics['avg_length'] = np.mean(lengths)
    metrics['avg_score'] = np.mean(scores)

    return metrics


def compute_metrics_destination_based(episode_stats, task):
    metrics = {}

    travel_distances = np.array(episode_stats['sum_travel_distance'])
    destination_reached = np.array(episode_stats['sum_destination_reached'])
    time_exceeded = np.array(episode_stats['sum_time_exceeded'])
    is_collision = np.array(episode_stats['sum_is_collision'])
    out_of_lane = np.array(episode_stats['sum_out_of_lane'])
    ttc = np.array(episode_stats['mean_ttc'])
    wpt_dis = np.array(episode_stats['mean_wpt_dis'])
    speed_norm = np.array(episode_stats['mean_speed_norm'])
    lengths = np.array(episode_stats['lengths'])
    scores = np.array(episode_stats['scores'])

    num_episodes = len(travel_distances)

    success_rate = np.mean(destination_reached)

    # Filter out failures after the tasks are successful
    is_collision[destination_reached > 0] = False
    time_exceeded[destination_reached > 0] = False
    out_of_lane[destination_reached > 0] = False

    metrics['task'] = task
    metrics['num_episodes'] = num_episodes
    metrics['success_rate'] = success_rate
    metrics['avg_travel_distance'] = np.mean(travel_distances)
    metrics['avg_destination_reached'] = np.mean(destination_reached)
    metrics['avg_destination_reached_ratio'] = np.sum(destination_reached) / num_episodes
    metrics['avg_time_exceeded'] = np.mean(time_exceeded)
    metrics['avg_time_exceeded_ratio'] = np.sum(time_exceeded) / num_episodes
    metrics['avg_is_collision'] = np.mean(is_collision)
    metrics['avg_is_collision_ratio'] = np.sum(is_collision) / num_episodes
    metrics['avg_out_of_lane'] = np.mean(out_of_lane)
    metrics['avg_out_of_lane_ratio'] = np.sum(out_of_lane) / num_episodes
    metrics['avg_ttc'] = np.mean(ttc)
    metrics['avg_wpt_dis'] = np.mean(wpt_dis)
    metrics['avg_speed_norm'] = np.mean(speed_norm)
    metrics['avg_length'] = np.mean(lengths)
    metrics['avg_score'] = np.mean(scores)

    return metrics


def main(args):
    task_success_distance_thresholds = {
        'carla_left_turn_simple': 43,
        'carla_left_turn_medium': 43,
        'carla_left_turn_hard': 43,
        'carla_right_turn_simple': 32,
        'carla_right_turn_medium': 32,
        'carla_right_turn_hard': 32,
        'carla_navigation': 50,
        'carla_lane_merge': 92,
        'carla_four_lane': 70,
        'carla_roundabout': 60,
        'carla_overtake': 20,
        'carla_message': 115,
    }

    jsonl_file = Path(args.logdir) / 'metrics.jsonl'
    task = args.task
    method = args.method

    episode_stats = {
        'sum_travel_distance': [],
        'sum_destination_reached': [],
        'sum_time_exceeded': [],
        'sum_is_collision': [],
        'sum_out_of_lane': [],
        'mean_ttc': [],
        'mean_wpt_dis': [],
        'mean_speed_norm': [],
        'lengths': [],
        'scores': []
    }

    with jsonl_file.open('r') as f:
        for line in f:
            data = json.loads(line)
            if 'stats/sum_travel_distance' in data:
                episode_stats['sum_travel_distance'].append(data['stats/sum_travel_distance'])
            if 'stats/sum_destination_reached' in data:
                episode_stats['sum_destination_reached'].append(data['stats/sum_destination_reached'])
            if 'stats/sum_time_exceeded' in data:
                episode_stats['sum_time_exceeded'].append(data['stats/sum_time_exceeded'])
            if 'stats/sum_is_collision' in data:
                episode_stats['sum_is_collision'].append(data['stats/sum_is_collision'])
            if 'stats/sum_out_of_lane' in data:
                episode_stats['sum_out_of_lane'].append(data['stats/sum_out_of_lane'])
            if 'stats/mean_ttc' in data:
                episode_stats['mean_ttc'].append(data['stats/mean_ttc'])
            if 'stats/mean_wpt_dis' in data:
                episode_stats['mean_wpt_dis'].append(data['stats/mean_wpt_dis'])
            if 'stats/mean_speed_norm' in data:
                episode_stats['mean_speed_norm'].append(data['stats/mean_speed_norm'])
            if 'episode/length' in data:
                episode_stats['lengths'].append(data['episode/length'])
            if 'episode/score' in data:
                episode_stats['scores'].append(data['episode/score'])

    if method == 'distance':
        final_metrics = compute_metrics_distance_based(episode_stats, task_success_distance_thresholds, task)
    elif method == 'destination':
        final_metrics = compute_metrics_destination_based(episode_stats, task)
    else:
        raise ValueError("Method should be either 'distance' or 'destination'")

    table = [
        ["Task", final_metrics['task']],
        ["Number of Episodes", final_metrics['num_episodes']],
        ["Success Rate", f"{final_metrics['success_rate']:.2%}"],
        ["Avg. Travel Distance", f"{final_metrics['avg_travel_distance']:.2f}"],
        ["Avg. Destination Reached", f"{final_metrics['avg_destination_reached']:.2f}"],
        ["Avg. Destination Reached Ratio", f"{final_metrics['avg_destination_reached_ratio']:.2%}"],
        ["Avg. Time Exceeded Ratio", f"{final_metrics['avg_time_exceeded_ratio']:.2%}"],
        ["Avg. Is Collision Ratio", f"{final_metrics['avg_is_collision_ratio']:.2%}"],
        ["Avg. Out of Lane Ratio", f"{final_metrics['avg_out_of_lane_ratio']:.2%}"],
        ["Avg. TTC", f"{final_metrics['avg_ttc']:.2f}"],
        ["Avg. Waypoint Distance", f"{final_metrics['avg_wpt_dis']:.2f}"],
        ["Avg. Speed Norm", f"{final_metrics['avg_speed_norm']:.2f}"],
        ["Avg. Episode Length", f"{final_metrics['avg_length']:.2f}"],
        ["Avg. Episode Score", f"{final_metrics['avg_score']:.2f}"],
    ]

    print("\nEvaluation Metrics:")
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Directory containing metrics.jsonl')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--method', type=str, required=True, choices=['distance', 'destination'], help='Method to compute success rate: distance or destination')
    args = parser.parse_args()

    main(args)
