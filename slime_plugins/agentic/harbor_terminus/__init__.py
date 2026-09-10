"""Harbor + Terminus-2 eval protocol.

API eval and local-ckpt Harbor eval both enter here. Scores are not comparable
to ECHO XML verifier rewards. Harbor talks to OpenAI HTTP; this package writes
the job YAML and evalkit dumps the trials.
"""

from .job_config import HarborJobSpec, build_config, write_job_config
from .images import (
    materialize_tasks_root,
    prepare_one,
    read_task_docker_image,
)

__all__ = [
    "HarborJobSpec",
    "build_config",
    "write_job_config",
    "materialize_tasks_root",
    "prepare_one",
    "read_task_docker_image",
]
