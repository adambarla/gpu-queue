# GPU Job Queue

A lightweight job queue for shared GPU servers without SLURM.

## Demo

<video src="demo.mov" controls title="Demo"></video>

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gpu-queue.git
    cd gpu-queue
    ```

2.  **Install with `uv` (Recommended)**:
    ```bash
    uv pip install -e .
    ```

## Usage

1.  **Start the Scheduler**:
    ```bash
    # Run in foreground (for testing/debugging)
    gpu-queue serve --min-free 2 --max-use 6

    # OR Start background daemon
    gpu-queue start --min-free 2 --max-use 6
    ```
    `--min-free` preserves that many physically idle GPUs. GPUs occupied by
    other users do not count toward this reserve. `--max-use` caps how many
    GPUs gpu-queue jobs may occupy at once.

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
    - `v`: Enter/leave select mode for bulk actions
    - `d`: **Duplicate** selected job into **Staging**
    - `n`: Create a **new staged** job
    - `e`: **Edit** selected staged job / **Save** staged changes
    - `s` or `Enter` (in Staging): Send staged job to **Pending** (with confirmation)
    - `b` (in Pending): Move selected pending job back to the top of **Staging**
    - `c`: **Discard** staged job, or cancel pending/running job
    - `J` / `K` (in Pending): Move selected job down/up in queue order
    - `Space`: View logs (internal viewer)
    - `L`: View logs in **external viewer** (`less`)
    - `p`: Pause/Resume running job
    - `r`: Retry completed job into **Staging**
    - `x`: Remove completed job

    In select mode, `j`/`k` extend the selected rows as you move. `Esc` clears the selection. Batch-safe commands apply to all selected rows in the active panel: `b`, `c`, `s`, `d`, `p`, `r`, `x`, and pending `J`/`K` reorder. Edit and logs remain cursor-only.

    **Interactive Editing**:
    - **Enter Edit Mode**: Press `e` on a staged job, or create one via `n`.
    - **Navigation**: Use `h`/`l` to switch between GPUs and Command fields.
    - **Modify Values**: Use `j`/`k` to decrease/increase GPU count.
    - **Edit Command**: Select the Command field and press `Enter` to open your system editor.
    - **Save**: Press `e` to save staged changes.
    - **Cancel**: Press `Esc` to discard changes.

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
