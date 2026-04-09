# Adding Benchmarks to AdaptFM

In addition to the existing segmentation benchmarks, AdaptFM also allows users to add new benchmarking algorithms. The steps are:

1. Navigate to AdaptFM > gui > widgets > metrics_widget.py
2. Create a new class that inherits from the "Metric" base class. Register it with MetricRegistry
3. Provide your class a 'name' and place your benchmarking algorithm in a 'compute' method. This method should take a parameter for 'ground_truth' files and 'prediction' files.
4. For examples look at current metrics in AdaptFM > gui > widgets > metrics_widget.py
