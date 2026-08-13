import argparse
import json
from AdaptFM.gui.widgets.metrics_widget import MetricRegistry
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric")
    parser.add_argument("--gt_dir")
    parser.add_argument("--models_json")
    parser.add_argument("--num_processes",type=int,default=None)
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    models_dirs = json.loads(args.models_json)

    metric = MetricRegistry.create(args.metric,args.num_processes)

    results = metric.compute( # dict[str, list[float]]
        args.gt_dir,
        models_dirs,
        args.output_dir    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{args.metric.lower().replace(' ', '_')}_results.csv"
    save_path = out_dir / filename

    df = pd.DataFrame(results)
    df.index.name = "sample_index"
    df.to_csv(save_path)

    print(f"Results saved to CSV: {save_path}")


if __name__ == "__main__":
    main()
