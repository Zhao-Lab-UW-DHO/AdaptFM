import argparse
import json
from AdaptFM.gui.widgets.metrics_widget import MetricRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric")
    parser.add_argument("--gt_dir")
    parser.add_argument("--models_json")
    parser.add_argument("--num_processes",type=int,default=None)

    args = parser.parse_args()

    models_dirs = json.loads(args.models_json)

    metric = MetricRegistry.create(args.metric,args.num_processes)

    metric.compute(
        args.gt_dir,
        models_dirs    )


if __name__ == "__main__":
    main()
