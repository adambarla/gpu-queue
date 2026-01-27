# GPU Job Queue

A lightweight job queue for shared GPU servers without SLURM.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gpu-queue.git
    cd gpu-queue-rs  # or relevant directory
    ```

2.  **Install with `uv` (Recommended)**:
    ```bash
    uv pip install -e .
    ```

## Usage

1.  **Start the Scheduler**:
    ```bash
    # Run in foreground (for testing/debugging)
    gpu-queue serve --min-free 2

    # OR Start background daemon
    gpu-queue start --min-free 2
    ```

2.  **Submit Jobs**:
    ```bash
    gpu-queue add "python train.py --config config.yaml" --gpus 4 --priority high
    gpu-queue add "bash scripts/eval.sh" --gpus 1 --front
    ```

3.  **Monitor & Manage (TUI)**:
    Open the interactive dashboard:
    ```bash
    gpu-queue watch
    ```
    **Keybindings**:
    - `d`: **Duplicate** selected job (enters Edit Mode)
    - `e`: **Edit** selected pending job / **Confirm** changes
    - `c`: **Cancel** job (Pending -> Cancelled status)
    - `Space`: View logs (internal viewer)
    - `L`: View logs in **external viewer** (`less`)
    - `p`: Pause/Resume running job
    - `r`: Remove completed job

## Development Setup

1.  **Install Dev Dependencies**:
    ```bash
    uv sync
    ```

2.  **Install Pre-commit Hooks** (Important for contributing):
    ```bash
    uv run pre-commit install
    ```
    This ensures code style checks run before every commit.
