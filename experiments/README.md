# Experiment framework: agents × environments

A single `config.json` describes a **comparison matrix** of
`environments × agents × seeds`. For every cell the runner trains the
agent, evaluates it, adds the configured baselines, and aggregates the
results across seeds into a table and a report.

```bash
python experiments/run.py experiments/configs/smoke.json            # full run
python experiments/run.py experiments/configs/smoke.json --dry-run  # validate + print plan only
python experiments/run.py experiments/configs/example_cologne.json --workers 4
python experiments/run.py experiments/configs/smoke.json --from-checkpoint output/checkpoints/smoke
```

See [docs/experiments.md](../docs/experiments.md) for the full reference:
config format, `defaults`/`overrides` keys, agent registry, checkpointing,
and the layout of the generated results.
