import json
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scheme.config import PathConfig
from scheme.ranker import Ranker


class Evaluator:
    """
    Evaluator is responsible for evaluating the performance of the RAG system.
    It calculates quality metrics, such as recall, and logs the results for analysis.
    """

    def __init__(self, path_config: PathConfig):
        """
        Initializes the Evaluator with the evaluation dataset and configuration.

        Args:
            path_config (PathConfig): Configuration object containing paths for evaluation data and logs.
        """

        self._config = path_config

        # Load the evaluation dataset from the specified path
        with open(self._config.eval_set_path) as f:
            self._test_data = [
                (test_pair["files"], test_pair["question"])
                for test_pair in json.load(f)
            ]


    # NOTE: Recall @ 10
    def _quality_metric(self, relevant: list[str], retrieved: list[str]) -> float:
        """
        Calculates the quality metric.

        Args:
            relevant (list[str]): List of relevant file paths.
            retrieved (list[str]): List of retrieved file paths.

        Returns:
            float: Quality metric score.
        """

        return sum(1 for e in relevant if e in retrieved) / len(relevant)

    def plot(self):
        """
        Generates a tradeoff plot based on cached run logs.
        """

        with open(self._config.logs_path) as f:
            logs = [line.strip().split(", ") for line in f.readlines()]
            slugs, quality, time = zip(*logs)

        labels = set(slugs)
        colors = plt.cm.tab20.colors
        label_to_color = {label: colors[i] for i, label in enumerate(labels)}

        plt.scatter(
            [float(q) for q in quality],
            [float(t) for t in time],
            c=[label_to_color[l] for l in slugs],
            alpha=0.5,
        )

        legend_elements = [
            Patch(facecolor=label_to_color[label], label=label)
            for label in labels
        ]
        plt.legend(
            handles=legend_elements,
            title="Approach",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
        )

        plt.xlabel("Quality (Recall @ 10)")
        plt.ylabel("Time (s)")
        plt.title("Tradeoff between Quality and Time")
        
        plt.tight_layout()
        plt.savefig(self._config.plot_path)


    async def test(self, ranker: Ranker, *, note: str = "RAG run", verbose: bool = False):
        """
        Tests the RAG system using the evaluation dataset and calculates metrics.

        Args:
            ranker (Ranker): The ranker used for retrieving relevant files.
            note (str): A note to include in the log file for this test run.
            verbose (bool): If True, prints detailed metrics for each query.

        Logs:
            Prints & writes the average quality and time metrics to the log file.
        """
        quality_sum, time_sum = 0.0, 0.0

        # TODO: progress
        for relevant, query in self._test_data:
            start_time = time.perf_counter()
            _, retrieved = await ranker.ainvoke(query)

            time_sum += time.perf_counter() - start_time
            quality_sum += self._quality_metric(relevant, retrieved)

            if verbose:
                print(f"Recall: {self._quality_metric(relevant, retrieved)}")

        quality_avg = quality_sum / len(self._test_data)
        time_avg = time_sum / len(self._test_data)

        # Log the results to the specified log file
        with open(self._config.logs_path, "a") as f:
            f.write(f"{note}, {quality_avg}, {time_avg}\n")

        print(f"\nMeasured metrics (query averaged)")
        print(f"Retrieval quality: {quality_avg} | Time: {time_avg}\n")


if __name__ == "__main__":
    from scheme.config import PathConfig

    config = PathConfig()
    eval = Evaluator(config)
    eval.plot()
