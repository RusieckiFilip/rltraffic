TARGET = "offline/transfer_curve.py"
TESTS = ["tests/test_p9_4_evaluation_review.py", "tests/test_p9_4_hooks.py"]
MUTANTS = [
    ("B-M13' grid4x4's subject admitted by both routes",
     '    registered = {*SUBJECTS, *SUBJECT_LAYOUT, ANCHOR_SUBJECT, GRID4X4_SUBJECT, *(row.subject for row in P7_3C_ARMS)}',
     '    registered = {*SUBJECTS, *SUBJECT_LAYOUT, ANCHOR_SUBJECT}'),
]
