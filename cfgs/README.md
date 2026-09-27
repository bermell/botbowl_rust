# Bot presets

A preset is an `MctsConfig` in TOML. It names only the knobs it changes; everything else comes
from the defaults compiled into the bot — **not** from `BLOOD_MCTS_*`, which a preset run ignores
outright. That is the point: a named configuration has to mean the same thing on every machine.

The file stem is the name, and the name is what gets stamped into `report.json`, the eval rung
label and a trajectory's provenance — so a result can always be traced back to what produced it.

```sh
# Same net, two configurations, head to head.
botbowl-ui eval --evaluator nn --model bbnet.onnx \
    --bot-config cfgs/aggressive.toml --vs-config cfgs/baseline.toml \
    --out report.json --per-game-out eval.games.jsonl
# → rungs named `mcts(nn:bbnet.onnx)@aggressive` vs `…@baseline`

# Generation under a named configuration; the name lands in each trajectory's `meta.extra`.
botbowl-ui dataset --mode random-start --bot-config cfgs/baseline.toml --out shard0.jsonl
```

`--bot-config` is exclusive with the per-knob flags (`--puct-mode`, `--backup`,
`--fpu-reduction`, `--horizon-turns` and their `--vs-` twins): a run is either fully described by
a preset or fully described by flags, never half of each. `--mcts-workers` is the exception — it
applies either way, because it is a property of the machine rather than of the bot.

`cfgs/baseline.toml` is the control: it sets nothing, so it *is* the shipped configuration, but it
gives the control arm a name that shows up in the report next to the variant's.

`budget_mode = "visits"` turns `--mcts-iters N` into "grow the root to N visits, at most N new
descents": a decision that inherits a big reused subtree then searches only the shortfall, rather
than N more on top of it (see `botbowl-mcts/CLAUDE.md`, "Budget mode"). `BLOOD_MCTS_BUDGET=visits`
is the environment spelling, which the hub pins into every job it submits.

Kickoff setup knobs (plan 047; see `botbowl-mcts/CLAUDE.md`, "Kickoff setup"):

```toml
setup = "auto"               # auto (search with a net, formation otherwise) | search | formation
opponent_setup = "auto"      # auto (= formation) | formation | search — the opponent model inside the tree
setup_formation = "line"     # line | spread | wedge | zone | random (drawn per drive)
setup_budget_scale = 1.0     # a setup decision's budget, as a multiple of the turn budget
setup_horizon_turns = 0      # own-turns of lookahead at a setup root; 0 = horizon_turns
```

`cfgs/setup-teacher.toml` is the gen-0 setup teacher: formation-driven placements recorded as
one-hot scripted samples.
