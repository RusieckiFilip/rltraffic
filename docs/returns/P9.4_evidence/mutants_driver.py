TARGET = "offline/campaigns/p9_4_mix50_fresh.sh"
TESTS = ["tests/test_p9_4_driver_review.py", "tests/test_p9_4_driver.py"]
MUTANTS = [
    ("D-B1 an unreadable state read as clean",
     'porcelain=$(git -C "$WORK_TREE" status --porcelain --ignored) \\\n  || refuse "git cannot read the state',
     'porcelain=$(git -C "$WORK_TREE" status --porcelain --ignored || true) \\\n  || refuse "git cannot read the state'),
    ("D-N9 ignored files invisible", 'status --porcelain --ignored)', 'status --porcelain)'),
    ("D-N8 any commit name", '[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] \\\n  || refuse', 'true \\\n  || refuse'),
    ("D-N7 a missing argument", '[ "$#" -eq 2 ] || refuse "usage:', 'true || refuse "usage:'),
    ("D-N5 noclobber off", 'set -o noclobber\n', ''),
    ("D-N9 bytecode written into the run tree", 'export PYTHONDONTWRITEBYTECODE=1\n', ''),
    ("D the module check fails open", '\\\n  || refuse "the P9.4 module does not import from $WORK_TREE"', ''),
    ("D on a branch admitted", '  0) refuse "$WORK_TREE is on a branch;', '  0) : refuse "$WORK_TREE is on a branch;'),
    ("D another commit admitted", '[ "$head" = "$COMMIT" ] || refuse', 'true || refuse'),
]
