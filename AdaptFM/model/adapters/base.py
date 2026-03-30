class DatasetAdapter:
    """
    Converts DatasetManager → model-specific training inputs.
    """

    def prepare(self, dataset_manager, output_dir):
        """
        Must return something consumable by training_command
        (path, dict, tuple, etc).
        """
        raise NotImplementedError
