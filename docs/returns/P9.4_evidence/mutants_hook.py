TARGET = "offline/transfer_curve.py"
TESTS = ["tests/test_p9_4_evaluation_review.py", "tests/test_p9_4_hooks.py"]
MUTANTS = [
    ("B-M13 grid4x4's subject admitted",
     '    registered = {*SUBJECTS, *SUBJECT_LAYOUT, ANCHOR_SUBJECT, GRID4X4_SUBJECT, *(row.subject for row in P7_3C_ARMS)}',
     '    registered = {*SUBJECTS, *SUBJECT_LAYOUT, ANCHOR_SUBJECT, *(row.subject for row in P7_3C_ARMS)}'),
    ("H1 P7.3c's arm subjects admitted", ', GRID4X4_SUBJECT, *(row.subject for row in P7_3C_ARMS)}', ', GRID4X4_SUBJECT}'),
    ("H2 any stage admitted",
     '    if not isinstance(stage, str) or not stage or stage in {*STAGES, STAGE_GRID4X4, *P7_3C_STAGES, STAGE_P7_3C}:',
     '    if False:'),
    ("H3 P7.3a's and P7.3b's stages admitted", 'stage in {*STAGES, STAGE_GRID4X4, *P7_3C_STAGES, STAGE_P7_3C}:',
     'stage in {STAGE_GRID4X4, *P7_3C_STAGES, STAGE_P7_3C}:'),
    ("H4 no stage admitted", '    if not isinstance(stage, str) or not stage or stage in',
     '    if isinstance(stage, str) and stage in'),
]
