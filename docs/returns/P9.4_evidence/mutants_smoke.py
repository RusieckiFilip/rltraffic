TARGET = "offline/p9_4_smoke.py"
TESTS = ["tests/test_p9_4_training_review.py", "tests/test_p9_4_smoke.py"]
MUTANTS = [
    ("N1s a symlink copied through", '        if target.is_symlink():\n            raise ValueError(f"{target} is a symlink;',
     '        if False:\n            raise ValueError(f"{target} is a symlink;'),
    ("N1s a directory outside", '        if parent != scratch_output and scratch_output not in parent.parents:', '        if False:'),
]
