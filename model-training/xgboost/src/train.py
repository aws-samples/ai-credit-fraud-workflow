import argparse
import os
import ray
from sagemaker_ray_helper import RayHelper
from ray.data import Dataset
from ray.train.xgboost import XGBoostTrainer
from ray.train import Result, ScalingConfig


def train_xgboost(
    num_workers: int,
    train_data: Dataset,
    test_data: Dataset,
    boosting_rounds: int = 100,
    use_gpu: bool = False,
) -> Result:

    params = {
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        label_column="TX_FRAUD_1",
        params=params,
       datasets={"train": train_data, "valid": test_data},
        # datasets={"train": train_data},
        num_boost_round=boosting_rounds,
    )
    result = trainer.fit()
    print(result.metrics)

    return result


if __name__ == "__main__":

    ray_helper = RayHelper()
    ray_helper.start_ray()

    parser = argparse.ArgumentParser()
    parser.add_argument("--boost_round", type=int, default=100)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)

    args = parser.parse_args()

    use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"

    if use_gpu:
        num_workers = int(ray.cluster_resources()["GPU"])
    else:
        num_workers = int(ray.cluster_resources()["CPU"]) - 2

    train_data = ray.data.read_parquet(args.train_data_path)
    test_data = ray.data.read_parquet(args.test_data_path)

    result = train_xgboost(
        num_workers, train_data, test_data, args.boost_round, use_gpu
    )

    model = XGBoostTrainer.get_model(result.checkpoint)

    model.save_model("/opt/ml/model/model.xgb")
