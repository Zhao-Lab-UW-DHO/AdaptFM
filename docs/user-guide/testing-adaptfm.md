# AdaptFM Demo Instructions

To help you get started, we have organized a curated set of sample datasets on [Hugging Face](https://huggingface.co/datasets/hbakhtiar/AdaptFM_Testing/tree/main). These datasets serve as an ideal way of testing and trying out AdaptFM on data known to work, allowing you to quickly verify your installation and explore the full platform workflow.

Once you have [installed AdaptFM](../getting-started/quick-start.md) (including the installation of any models you plan to use via the Model Installer) and downloaded the demo datasets, you can familiarize yourself with AdaptFM through the workflow described in our [user guide overview](user-guide.md)

## Download Instructions
Use 'include' to specify the dataset you want to download. The folder name should have /* at the end to download all contents in the folder. The folder name should be in quotes as below.  

**The entire test dataset repository is over 100 GB. Make sure you have enough disk space before downloading everything at once, or download only specific dataset folders as needed.**

* **Download a specific dataset folder (e.g., BBBC024):**
    ```bash
    hf download hbakhtiar/AdaptFM_Testing --repo-type dataset --include "BBBC024/*" --local-dir "/path/to/your/folder"
    ```

* **Download the entire test dataset (~100 GB):**
    ```bash
    hf download hbakhtiar/AdaptFM_Testing --repo-type dataset --local-dir "/path/to/your/folder"
    ```

* **Download the SSVT Organoids model checkpoint:**
    ```bash
    hf download hbakhtiar/SSVT_Organoids --local-dir "/path/to/your/folder"
    ```

The datasets are structured to follow the standard input expectations described in the [User Guide](user-guide.md).