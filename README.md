# ERA-Bench

ERA-Bench is a project for embodied Abstract Reasoning Corpus (ARC-AGI-1)
tasks. Its simulation suite, **ERA-Bench Sim**, generates the delivered
**ARC-Embodied** dataset. ERA-Bench Sim turns a symbolic ARC input/output pair
into a tabletop manipulation problem: a Franka robot observes the example
grid, selects colored cubes from reserve stacks, and constructs the target
output grid.

This repository provides ERA-Bench Sim and the ARC-Embodied dataset-generation
pipeline. It is **not an RL training repository**. The included benchmark policy is a
deterministic state machine backed by differential inverse kinematics and
cuRobo motion planning. Because the environment uses Isaac Lab's manager-based
interfaces, it can be extended with reward terms, agent configurations, and a
training runner if reinforcement learning is desired.

## Task source and citation

ERA-Bench does **not** introduce new abstract-reasoning tasks. It uses the
original ARC-AGI-1 task corpus, retaining its task IDs and input/output
examples while expressing each output grid as a physical cube-manipulation
episode. Cite the original ARC benchmark when using ERA-Bench:

> François Chollet. 2019. *On the Measure of Intelligence.* arXiv:1911.01547
> [cs.AI]. https://doi.org/10.48550/arXiv.1911.01547

```bibtex
@article{chollet2019measure,
  title={On the Measure of Intelligence},
  author={Chollet, Fran\c{c}ois},
  journal={arXiv preprint arXiv:1911.01547},
  year={2019},
  doi={10.48550/arXiv.1911.01547}
}
```

See the [paper](https://arxiv.org/abs/1911.01547) and the official
[ARC-AGI-1 benchmark page](https://arcprize.org/arc-agi/1) for the original
task corpus.

## Evaluation benchmark

The companion [ERA-Bench Evaluation](https://github.com/nardo7/era_bench_evaluation)
repository contains the benchmark implementation for ARC-Embodied. Use that
repository to evaluate models on the ARC-Embodied dataset; this repository
provides ERA-Bench Sim and the dataset-generation pipeline.

## Start with the dataset

The ARC-Embodied HDF5 dataset is the main artifact. Most users can
use the dataset directly and do not need to install Isaac Sim, Isaac
Lab, or run this repository.

Each successful episode can contain:

- the ARC task ID, task-local pair index, and train/test membership;
- robot actions, states, and structured observations;
- RGB, depth, and colorized semantic-segmentation observations from the table
  and wrist cameras, plus an overhead RGB view;
- a high-level manipulation timeline encoding grasp and place operations.

See the [ARC-Embodied dataset reference](source/era_bench/docs/arc_embodied_dataset.md)
for the HDF5 layout, split definitions, array shapes, and recording cadence.

Dataset files are intentionally not tracked by Git. Place downloaded HDF5
artifacts under `datasets/` when using the included inspection tools.

For example, to export table-camera RGB observations as MP4 files without
starting Isaac Sim:

```bash
python scripts/dataset/create_videos.py \
    --dataset datasets/arc_embodied.hdf5 \
    --fps 20
```

The command requires Python packages for HDF5 and video handling (`h5py`,
`imageio`, and an FFmpeg backend), but not the simulator. Use `--task` and
`--pair` to export only one ARC task or pair, and `--data-type` to select RGB,
depth, or semantic segmentation.

The recording cadence is `20 Hz`. The example passes `--fps 20` to preserve
that timing; the export tool's default playback rate is `30 fps`.

## ERA-Bench Sim overview

ERA-Bench Sim uses the registered environment `ARC-Embodied-Era-Bench-v0`.
A simulation run combines:

- an Isaac Lab manager-based environment and ARC-specific scene;
- a Franka Panda robot, input/output grids, reserve cube stacks, and three
  cameras;
- a state machine that produces high-level pick-and-place commands;
- IK monitoring and recovery logic;
- a hybrid action term that switches between differential IK and cuRobo;
- task scheduling, reset events, success/failure termination, and HDF5
  recording.

The input grid is a visual reference. The robot manipulates reserve cubes to
realize the requested output grid. Successful and failed attempts are exported
separately, allowing interrupted generation runs to resume from existing
successful task-pair coverage.

## Setup for simulation development

Only follow this section if you want to reproduce the dataset, modify the
environment, or build a new policy on top of it.

Requirements:

- Linux with a supported NVIDIA GPU and driver;
- Isaac Sim and Isaac Lab (the current development environment uses Isaac Sim
  5.1);
- the Isaac Lab Mimic and cuRobo dependencies;
- Git with submodule support.

Clone the ERA-Bench Sim repository and initialize its `arc_utils` dependency:

```bash
git clone --recurse-submodules https://github.com/nardo7/era_bench.git
cd era_bench
git submodule update --init --recursive
```

Install the extension with the Python environment managed by Isaac Lab:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
"$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e source/era_bench
```

Do not use an unrelated system Python for simulator commands. On a cluster,
run the same commands inside the site's GPU-enabled Isaac Lab Apptainer image.

## Quick start

Run the test suite:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p -m pytest \
    source/era_bench/test \
    --import-mode=importlib \
    -v
```

Run a short headless simulation and write to a new smoke-test dataset:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p \
    scripts/benchmark/arc_state_machine_agent.py \
    --task ARC-Embodied-Era-Bench-v0 \
    --headless \
    --num_envs 1 \
    --max_steps 1000 \
    --dataset_file datasets/smoke_test.hdf5
```

This is a simulator smoke test, not a command for reproducing the complete
ARC-Embodied dataset. Full generation requires substantially longer runs and
sufficient storage for camera observations. By default, an existing dataset is
opened in append/resume mode. Pass `--overwrite_dataset` only when you
intentionally want to replace the selected output file.

Useful runner options include:

- `--num_envs`: number of parallel simulated environments;
- `--max_steps`: maximum environment steps before exit;
- `--store_videos`: additionally store episode videos;
- `--debug`: print state-machine and target-pose diagnostics;
- `--track`: enable optional Weights & Biases progress logging.

## Repository layout

```text
arc_utils/                         ARC task, grid, and color utilities
scripts/
  benchmark/                       simulator runner and runtime logging
  dataset/                         HDF5 inspection and video-export tools
source/era_bench/
  config/                          Isaac Lab extension metadata
  docs/architecture/               detailed runtime and data-flow documentation
  era_bench/
    tasks/manager_based/
      arc_embodied/
        control/                    state machine and IK supervision
        env/                        environment, scene, and configuration
        mdp/                        actions, commands, events, observations,
                                    terminations, and recorders
      common/                       shared dataset, geometry, and recorder tools
  test/arc_embodied/               simulation-aware unit tests
```

The architecture documents provide deeper views of the
[overall ARC integration](source/era_bench/docs/architecture/arc_embodied_architecture.md),
[runtime layers](source/era_bench/docs/architecture/arc_embodied_runtime_layers.md),
and [recorded high-level actions](source/era_bench/docs/architecture/arc_embodied_high_level_actions.md).

## Extending ERA-Bench Sim to reinforcement learning

The environment already exposes the main Isaac Lab integration points needed
by an RL workflow: action terms, observation groups, reset events, termination
terms, and vectorized environments. An RL extension would still need to define
an explicit learning objective and training stack—most importantly reward
terms, policy observations, an agent configuration, and a training/evaluation
entry point. Those pieces are intentionally absent from this repository's
current simulation-and-dataset scope.

## Scaling grid sizes

The current `8 × 8` limit is primarily a physical-workspace constraint: the
robot must reach every reserve and output pose with a valid grasp/place pose.
Larger grids are possible in principle with a longer-reach arm and coordinated
updates to:

- **Task selection:** in `arc_utils/src/arc_utils/analysis.py`, change the
  `max_size=8` argument in `get_era_bench_tasks()`. If the expanded task set
  needs more colors or cubes, also change the calls that currently limit output
  grids to four colors and 12 cells per color.
- **Grid geometry and capacity:** in
  [`scene_cfg.py`](source/era_bench/era_bench/tasks/manager_based/arc_embodied/env/scene_cfg.py),
  update `max_grid_size`, `cubes_size`, `separation_distance`,
  `input_grid_center_local`, and `output_grid_center_local`. Increase
  `input_grid_cubes_amount_per_color`, `input_grid_blue_cubes_amount`,
  `reserve_stacks_amount`, `reserve_cubes_amount_per_stack`, and
  `reserve_cubes_amount_per_line` as required, then reposition the reserve
  stacks and resize the table.
- **Runtime buffers:** in
  [`commands.py`](source/era_bench/era_bench/tasks/manager_based/arc_embodied/mdp/commands.py),
  increase `MAX_OBJECTS_PER_GRID` and `MAX_OUTPUT_OBJECTS_PER_GRID` to cover the
  largest non-background input and output. `environment.py` derives its target
  buffers from `MAX_OUTPUT_OBJECTS_PER_GRID`.
- **Robot and planning:** replace `EraBenchSceneCfg.robot` and update
  `FRANKA_GRASP_CENTER_OFFSET`. In
  [`env_cfg.py`](source/era_bench/era_bench/tasks/manager_based/arc_embodied/env/env_cfg.py),
  update the arm/gripper joint names, hand body name, default joint pose,
  end-effector frame paths, wrist-camera parent prim, gripper commands, and
  `curobo_task_name`. Supply a matching cuRobo robot configuration and update
  `_MAX_TASK_TRAJECTORY_DISTANCE` in
  [`curobo_planner_extended.py`](source/era_bench/era_bench/curobo_planner_extended.py);
  rescale `_SHORT_PATH_DISTANCE` if the complete physical scale changes.
- **Workspace scale:** resize/reposition the table and reserve area and update `scene.env_spacing`
  plus `displacement_threshold`, `min_step_position_delta`, success
  `distance_threshold`, state-machine `position_threshold`, and controller
  `hold_position_tolerance`. If cube size changes, also rescale cube `mass`,
  `contact_offset`, `rest_offset`, and PhysX `friction_correlation_distance`.

> **Warning:** Making the cubes smaller is likely to affect AI-model performance
> dramatically. Treat data generated at another physical scale as a separate
> benchmark configuration.
