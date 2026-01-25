# GPU Job Queue

A lightweight job queue for shared GPU servers without SLURM.

## Usage

1.  **Start the daemon (optional, serves in background)**:
    ```bash
    gpu-queue start --min-free 2
    ```
    Or run in foreground:
    ```bash
    gpu-queue serve --min-free 2
    ```

    You can exclude specific GPUs:
    ```bash
    gpu-queue serve --exclude-gpus 1,2
    ```

2.  **Add jobs**:
    ```bash
    gpu-queue add "python train.py" --gpus 1
    ```

3.  **Monitor**:
    ```bash
    gpu-queue watch
    ```

4.  **Manage**:
    ```bash
    gpu-queue cancel <job_id>
    gpu-queue retry <job_id>
    gpu-queue pause <job_id>
    gpu-queue status # Check if running
    ```
    ```

## Installation

```bash
uv pip install -e .
```

## Development

To set up the development environment with pre-commit hooks:

1.  **Install development dependencies**:
    ```bash
    uv sync
    ```

2.  **Install pre-commit hooks**:
    ```bash
    uv run pre-commit install
    ```

