import json
import time

from scheme.config import PathConfig
from scheme.ranker import Ranker


class Evaluator:
    """
    TOOD: docstring
    """

    def __init__(self, path_config: PathConfig):
        self._config = path_config
        self._config.logs_path.touch()
        with open(self._config.eval_set_path) as f:
            self._test_data = json.load(f)


    # NOTE: Recall @ 10
    def _quality_metric(self, relevant: list[str], retrieved: list[str]) -> float:
        return sum(1 for e in relevant if e in retrieved) / len(relevant)


    # TODO: Generate tradeoff plot, based on cached run logs
    def plot(self):
        pass


    async def test(self, ranker: Ranker, *, note: str = "RAG run", verbose: bool = False):
        quality_sum, time_sum = 0.0, 0.0
        for relevant, query in self._test_data:
            start_time = time.perf_counter()
            retrieved = await ranker.ainvoke(query)

            time_sum += time.perf_counter() - start_time
            quality_sum += self._quality_metric(relevant, retrieved)

            if verbose:
                print(self._quality_metric(relevant, retrieved))

        quality_avg = quality_sum / len(self._test_data)
        time_avg = time_sum / len(self._test_data)

        with open(self._config.logs_path, "a") as f:
            f.write(f"{note}, {quality_avg}, {time_avg}")

        print(f"Measured metrics (query averaged)")
        print(f"Retrieval quality: {quality_avg} | Time: {time_avg}\n")


if __name__ == "__main__":
    pass
