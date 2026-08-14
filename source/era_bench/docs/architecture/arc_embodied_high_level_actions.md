# ARC Embodied High-Level Actions

## Stored structure

Each recorded ARC episode contains one high-level action timeline:

```text
/data
└── demo_<n>
    └── high_level_actions
        └── action_timeline    int64 [1, T, 5]
```

`T` is the number of timesteps in the recorded episode. An entry is addressed
as:

```text
action_timeline[0, t] = [phase, color_id, location_type, row, col]
```

| Field | Stored meaning |
|---|---|
| `phase` | `0` = no action, `1` = grasp, `2` = place |
| `color_id` | Canonical ARC color ID of the selected cube; `-1` when `phase` is `0` |
| `location_type` | `0` = reserve, `1` = output grid, `-1` = no action |
| `row` | Zero-based output-grid row for a place; `-1` for a reserve grasp or no action |
| `col` | Zero-based output-grid column for a place; `-1` for a reserve grasp or no action |

Rows and columns use the original ARC output-grid coordinate system, with `(0, 0)` at the top-left. They are derived from the observed cube pose when placement completes, not from the planned goal order. The current simulation records reserve grasps as `[1, color_id, 0, -1, -1]` and output-grid placements as `[2, color_id, 1, row, col]`.

The leading dimension is a singleton because the complete timeline is stored
once as episode-level recorder data.

## Diagram

![ARC Embodied stored high-level actions](arc_embodied_high_level_actions.svg)

Editable Graphviz source:
[`arc_embodied_high_level_actions.dot`](arc_embodied_high_level_actions.dot)
