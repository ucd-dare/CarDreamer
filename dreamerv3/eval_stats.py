import argparse
import json
from pathlib import Path

import numpy as np
from tabulate import tabulate


def compute_metrics_destination_based(episode_stats, distance_threshold=None):
    metrics = {}

    travel_distances = np.array(episode_stats["sum_travel_distance"])
    destination_reached = np.array(episode_stats["sum_destination_reached"])
    time_exceeded = np.array(episode_stats["sum_time_exceeded"])
    is_collision = np.array(episode_stats["sum_is_collision"])
    out_of_lane = np.array(episode_stats["sum_out_of_lane"])
    ttc = np.array(episode_stats["mean_ttc"])
    wpt_dis = np.array(episode_stats["mean_wpt_dis"])
    speed_norm = np.array(episode_stats["mean_speed_norm"])
    lengths = np.array(episode_stats["lengths"])
    scores = np.array(episode_stats["scores"])

    num_episodes = len(travel_distances)
    batch_size = num_episodes // 3

    def compute_mean_and_sem(data):
        batch_means = [np.mean(data[i * batch_size : (i + 1) * batch_size]) for i in range(3)]
        mean = np.mean(batch_means)
        sem = np.std(batch_means, ddof=1) / np.sqrt(3)
        return mean, sem

    if distance_threshold is not None:
        metrics["success_rate"] = np.mean(travel_distances >= distance_threshold)
    else:
        metrics["success_rate"] = np.mean(destination_reached)

    is_collision[destination_reached > 0] = False
    time_exceeded[destination_reached > 0] = False
    out_of_lane[destination_reached > 0] = False

    metrics["num_episodes"] = num_episodes
    metrics["success_rate"], metrics["success_rate_sem"] = compute_mean_and_sem(
        travel_distances >= distance_threshold if distance_threshold is not None else destination_reached
    )
    (
        metrics["avg_travel_distance"],
        metrics["avg_travel_distance_sem"],
    ) = compute_mean_and_sem(travel_distances)
    (
        metrics["avg_destination_reached"],
        metrics["avg_destination_reached_sem"],
    ) = compute_mean_and_sem(destination_reached)
    (
        metrics["avg_time_exceeded"],
        metrics["avg_time_exceeded_sem"],
    ) = compute_mean_and_sem(time_exceeded)
    metrics["avg_is_collision"], metrics["avg_is_collision_sem"] = compute_mean_and_sem(is_collision)
    metrics["avg_out_of_lane"], metrics["avg_out_of_lane_sem"] = compute_mean_and_sem(out_of_lane)
    metrics["avg_ttc"], metrics["avg_ttc_sem"] = compute_mean_and_sem(ttc)
    metrics["avg_wpt_dis"], metrics["avg_wpt_dis_sem"] = compute_mean_and_sem(wpt_dis)
    metrics["avg_speed_norm"], metrics["avg_speed_norm_sem"] = compute_mean_and_sem(speed_norm)
    (
        metrics["avg_speed_over_4_ratio"],
        metrics["avg_speed_over_4_ratio_sem"],
    ) = compute_mean_and_sem(speed_norm / 4.0)
    metrics["avg_length"], metrics["avg_length_sem"] = compute_mean_and_sem(lengths)
    metrics["avg_score"], metrics["avg_score_sem"] = compute_mean_and_sem(scores)

    return metrics


def main(args):
    jsonl_file = Path(args.logdir) / "metrics.jsonl"

    episode_stats = {
        "sum_travel_distance": [],
        "sum_destination_reached": [],
        "sum_time_exceeded": [],
        "sum_is_collision": [],
        "sum_out_of_lane": [],
        "mean_ttc": [],
        "mean_wpt_dis": [],
        "mean_speed_norm": [],
        "lengths": [],
        "scores": [],
    }

    with jsonl_file.open("r") as f:
        for line in f:
            data = json.loads(line)
            if "stats/sum_travel_distance" in data:
                episode_stats["sum_travel_distance"].append(data["stats/sum_travel_distance"])
            if "stats/sum_destination_reached" in data:
                episode_stats["sum_destination_reached"].append(data["stats/sum_destination_reached"])
            if "stats/sum_time_exceeded" in data:
                episode_stats["sum_time_exceeded"].append(data["stats/sum_time_exceeded"])
            if "stats/sum_is_collision" in data:
                episode_stats["sum_is_collision"].append(data["stats/sum_is_collision"])
            if "stats/sum_out_of_lane" in data:
                episode_stats["sum_out_of_lane"].append(data["stats/sum_out_of_lane"])
            if "stats/mean_ttc" in data:
                episode_stats["mean_ttc"].append(data["stats/mean_ttc"])
            if "stats/mean_wpt_dis" in data:
                episode_stats["mean_wpt_dis"].append(data["stats/mean_wpt_dis"])
            if "stats/mean_speed_norm" in data:
                episode_stats["mean_speed_norm"].append(data["stats/mean_speed_norm"])
            if "episode/length" in data:
                episode_stats["lengths"].append(data["episode/length"])
            if "episode/score" in data:
                episode_stats["scores"].append(data["episode/score"])

    final_metrics = compute_metrics_destination_based(episode_stats, distance_threshold=args.distance)

    table = [
        ["Number of Episodes", final_metrics["num_episodes"]],
        [
            "Success Rate",
            f"{final_metrics['success_rate']:.2%} ± {final_metrics['success_rate_sem']:.2%}",
        ],
        [
            "Avg. Travel Distance",
            f"{final_metrics['avg_travel_distance']:.2f} ± {final_metrics['avg_travel_distance_sem']:.2f}",
        ],
        [
            "Avg. Destination Reached",
            f"{final_metrics['avg_destination_reached']:.2f} ± {final_metrics['avg_destination_reached_sem']:.2f}",
        ],
        [
            "Avg. Time Exceeded",
            f"{final_metrics['avg_time_exceeded']:.2f} ± {final_metrics['avg_time_exceeded_sem']:.2f}",
        ],
        [
            "Avg. Is Collision",
            f"{final_metrics['avg_is_collision']:.2%} ± {final_metrics['avg_is_collision_sem']:.2%}",
        ],
        [
            "Avg. Out of Lane",
            f"{final_metrics['avg_out_of_lane']:.2%} ± {final_metrics['avg_out_of_lane_sem']:.2%}",
        ],
        [
            "Avg. TTC",
            f"{final_metrics['avg_ttc']:.2f} ± {final_metrics['avg_ttc_sem']:.2f}",
        ],
        [
            "Avg. Waypoint Distance",
            f"{final_metrics['avg_wpt_dis']:.2f} ± {final_metrics['avg_wpt_dis_sem']:.2f}",
        ],
        [
            "Avg. Speed Norm",
            f"{final_metrics['avg_speed_norm']:.2f} ± {final_metrics['avg_speed_norm_sem']:.2f}",
        ],
        [
            "Avg. Speed Over 4 Ratio",
            f"{final_metrics['avg_speed_over_4_ratio']:.2%} ± {final_metrics['avg_speed_over_4_ratio_sem']:.2%}",
        ],
        [
            "Avg. Episode Length",
            f"{final_metrics['avg_length']:.2f} ± {final_metrics['avg_length_sem']:.2f}",
        ],
        [
            "Avg. Episode Score",
            f"{final_metrics['avg_score']:.2f} ± {final_metrics['avg_score_sem']:.2f}",
        ],
    ]

    print("\nEvaluation Metrics:")
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="Directory containing metrics.jsonl")
    parser.add_argument("--distance", type=float, help="Distance threshold for calculating success rate")
    args = parser.parse_args()

    main(args)
