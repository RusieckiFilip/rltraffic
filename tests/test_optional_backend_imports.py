"""Optional simulator backends should stay import-lazy."""


def test_moss_env_public_import_does_not_require_engines():
    from envs import MossEnv
    from envs.moss_env import DEFAULT_JUNCTION_YELLOW_TIME

    assert MossEnv.__name__ == "MossEnv"
    assert DEFAULT_JUNCTION_YELLOW_TIME == 0.0
