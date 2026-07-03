"""Config-driven experiment framework.

Compare traffic-signal agents across multiple simulator environments
(CityFlow / SUMO / MOSS) from a single JSON config that describes a
matrix of ``environments x agents x seeds``.

Entry point: ``python experiments/run.py <config.json>``.
See ``docs/experiments.md`` for the config schema.
"""
