from pathlib import Path

import environ

env = environ.Env()

REPO_ROOT = Path(__file__).resolve().parents[5]
env_file = REPO_ROOT / "envs" / "dev" / ".env.template"
if not env_file.exists():
    env_file = REPO_ROOT / "app" / "envs" / "dev" / ".env.template"

if env_file.exists():
    env.read_env(env_file=env_file)
else:
    raise RuntimeError(f"Couldn't find .env.template at {env_file!r}")

from project.settings import *  # noqa: E402,F403
