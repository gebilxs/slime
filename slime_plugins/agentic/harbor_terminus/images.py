"""Prepare local Docker images so Harbor can use force_build=false.

TB2 task.toml declares ``docker_image = "alexgshaw/<task>:20251031"``.
GPFS tars often load as ``registry-v2.h.pjlab.org.cn/ailab/terminal-bench2:...``.
This module tags the loaded image to the name Harbor expects.

Uses infra.docker.runtime so login tests inject FakeDocker. No dockerd required
for dry-run / already-present / load-and-tag unit tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from infra.docker.runtime import DockerRuntime, SubprocessDockerRuntime

# PJLab apt proxy. Terminus needs tmux; most TB2 images do not ship it.
_BAKE_TMUX_SCRIPT = r"""
set -eux
. /etc/os-release
id="${ID:-unknown}"
suite="${VERSION_CODENAME:-}"
rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

install_from_suite() {
  use_suite=$1
  insecure=$2
  base="http://mirrors.i.h.pjlab.org.cn/repository/apt-${use_suite}-proxy/ubuntu"
  cat >/etc/apt/sources.list <<EOF
deb $base $use_suite main universe
deb $base $use_suite-updates main universe
deb $base $use_suite-security main universe
EOF
  if [ "$insecure" = "1" ]; then
    apt-get -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update -qq
    DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::AllowInsecureRepositories=true -o APT::Get::AllowUnauthenticated=true --allow-unauthenticated install -y -qq tmux
  else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
  fi
  tmux -V
}

if [ "$id" = "ubuntu" ] && [ "$suite" = "noble" ]; then
  install_from_suite noble 0 || install_from_suite noble 1 || install_from_suite jammy 1
elif [ "$id" = "ubuntu" ] && [ -n "$suite" ]; then
  install_from_suite "$suite" 1 || install_from_suite jammy 1 || install_from_suite noble 1
else
  install_from_suite jammy 1 || install_from_suite noble 1
fi
"""


def inject_docker_image(toml_text: str, image: str) -> str:
    """Add docker_image to task.toml when Harbor needs a preloaded name.

    TBLite tasks often only have a Dockerfile and no docker_image field.
    """
    if re.search(r"^\s*docker_image\s*=", toml_text, re.MULTILINE):
        return toml_text
    block = f'docker_image = "{image}"\n'
    if "[environment]" in toml_text:
        return toml_text.replace("[environment]", "[environment]\n" + block, 1)
    return toml_text.rstrip() + "\n\n[environment]\n" + block


def materialize_task(src: Path, dst: Path, image: str | None) -> Path:
    """Writable task dir for Harbor: symlink contents, patch task.toml."""
    import os

    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "task.toml":
            continue
        if target.exists() or target.is_symlink():
            continue
        os.symlink(child.resolve(), target)
    toml_src = src / "task.toml"
    text = toml_src.read_text(encoding="utf-8", errors="replace") if toml_src.is_file() else ""
    if image:
        text = inject_docker_image(text, image)
    (dst / "task.toml").write_text(text, encoding="utf-8")
    return dst


def materialize_tasks_root(
    src_root: Path,
    dst_root: Path,
    *,
    mmap: dict[str, str] | None = None,
    tasks: list[str] | None = None,
) -> Path:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    mmap = mmap or {}
    dirs = [src_root / t for t in tasks] if tasks else sorted(
        d for d in src_root.iterdir() if d.is_dir() and (d / "task.toml").is_file()
    )
    for src in dirs:
        if not src.is_dir():
            continue
        image = read_task_docker_image(src) or mmap.get(src.name)
        materialize_task(src, dst_root / src.name, image)
    return dst_root


def read_task_docker_image(task_dir: str | Path) -> str | None:
    toml = Path(task_dir) / "task.toml"
    if not toml.is_file():
        return None
    in_env = False
    for raw in toml.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_env = line == "[environment]"
            continue
        if in_env and line.startswith("docker_image"):
            _, _, rhs = line.partition("=")
            return rhs.strip().strip('"').strip("'")
    return None


def manifest_map(path: str | Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if path is None:
        return out
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        img = row.get("image") or row.get("image_ref")
        if not img:
            continue
        tid = str(row.get("task_id") or row.get("task_name") or "")
        tp = str(row.get("task_path") or "")
        if tid:
            out[tid] = img
        if tp:
            out[Path(tp).name] = img
    return out


def _run(runtime: DockerRuntime, argv: list[str], timeout: int = 120) -> tuple[int, str, str]:
    return runtime.run(argv, timeout)


def image_present(runtime: DockerRuntime, name: str) -> bool:
    code, _, _ = _run(runtime, ["docker", "image", "inspect", name], 30)
    return code == 0


def image_has_tmux(runtime: DockerRuntime, name: str) -> bool:
    code, _, _ = _run(
        runtime,
        ["docker", "run", "--rm", "--network", "none", name, "bash", "-lc", "tmux -V"],
        60,
    )
    return code == 0


def load_tar(runtime: DockerRuntime, tar_path: Path, timeout: int = 900) -> None:
    code, out, err = _run(runtime, ["docker", "load", "-i", str(tar_path)], timeout)
    if code != 0:
        raise RuntimeError(f"docker load failed: {(err or out)[-400:]}")


def tag_image(runtime: DockerRuntime, src: str, dst: str) -> None:
    code, out, err = _run(runtime, ["docker", "tag", src, dst], 60)
    if code != 0:
        raise RuntimeError(f"docker tag {src} {dst} failed: {(err or out)[-300:]}")


def bake_tmux(runtime: DockerRuntime, name: str) -> str:
    """Install tmux via PJLab apt mirror and commit onto the same tag."""
    if image_has_tmux(runtime, name):
        return "already_present"
    cid = f"harbor-bake-tmux-{abs(hash(name)) % 10_000_000}"
    _run(runtime, ["docker", "rm", "-f", cid], 30)
    code, out, err = _run(
        runtime,
        ["docker", "run", "-d", "--name", cid, "--network", "host", name, "sleep", "infinity"],
        120,
    )
    if code != 0:
        raise RuntimeError(f"bake tmux run failed: {(err or out)[-400:]}")
    try:
        code, out, err = _run(runtime, ["docker", "exec", cid, "bash", "-lc", _BAKE_TMUX_SCRIPT], 600)
        if code != 0:
            raise RuntimeError(f"bake tmux failed for {name}: {(err or out)[-500:]}")
        code, out, err = _run(runtime, ["docker", "commit", cid, name], 120)
        if code != 0:
            raise RuntimeError(f"docker commit failed: {(err or out)[-300:]}")
    finally:
        _run(runtime, ["docker", "rm", "-f", cid], 30)
    if not image_has_tmux(runtime, name):
        raise RuntimeError(f"tmux still missing after bake: {name}")
    return "baked"


def prepare_one(
    task_dir: Path,
    *,
    runtime: DockerRuntime,
    mmap: dict[str, str],
    tar_root: Path | None,
    dry_run: bool = False,
    pull: bool = False,
    bake: bool = False,
) -> dict[str, Any]:
    tid = task_dir.name
    expected = read_task_docker_image(task_dir)
    src = mmap.get(tid)
    rec: dict[str, Any] = {
        "task_id": tid,
        "expected": expected,
        "source": src,
        "status": "skip",
    }
    if not expected:
        rec["status"] = "no_docker_image_in_task_toml"
        return rec

    if image_present(runtime, expected):
        rec["status"] = "already_present"
    else:
        tar = None
        if tar_root is not None:
            cand = tar_root / tid / "image.tar"
            if cand.is_file():
                tar = cand
        if dry_run:
            rec["status"] = "would_prepare"
            rec["tar"] = str(tar) if tar else None
            return rec
        if tar is not None:
            load_tar(runtime, tar)
            if src and image_present(runtime, src):
                tag_image(runtime, src, expected)
            elif not image_present(runtime, expected):
                meta = tar.parent / "meta.json"
                if meta.is_file():
                    loaded = json.loads(meta.read_text()).get("image")
                    if loaded and image_present(runtime, loaded):
                        tag_image(runtime, loaded, expected)
            if not image_present(runtime, expected) and src and image_present(runtime, src):
                tag_image(runtime, src, expected)
            if not image_present(runtime, expected):
                rec["status"] = "load_ok_but_tag_missing"
                rec["tar"] = str(tar)
                return rec
            rec["status"] = "loaded_and_tagged"
            rec["tar"] = str(tar)
        elif src and pull:
            code, out, err = _run(runtime, ["docker", "pull", src], 600)
            if code != 0:
                rec["status"] = "pull_failed"
                rec["error"] = (err or out or "")[-300:]
                return rec
            tag_image(runtime, src, expected)
            rec["status"] = "pulled_and_tagged"
        else:
            rec["status"] = "missing_tar_and_no_pull"
            return rec

    if bake and not dry_run and image_present(runtime, expected):
        try:
            rec["tmux"] = bake_tmux(runtime, expected)
        except Exception as exc:  # noqa: BLE001
            rec["tmux"] = "bake_failed"
            rec["tmux_error"] = str(exc)[-400:]
    return rec


def prepare_tasks(
    tasks_root: Path,
    *,
    runtime: DockerRuntime | None = None,
    manifest: Path | None = None,
    tar_root: Path | None = None,
    tasks: list[str] | None = None,
    dry_run: bool = False,
    pull: bool = False,
    bake: bool = False,
) -> dict[str, Any]:
    runtime = runtime or SubprocessDockerRuntime()
    mmap = manifest_map(manifest)
    if tasks:
        dirs = [tasks_root / t for t in tasks]
    else:
        dirs = sorted(
            d for d in tasks_root.iterdir() if d.is_dir() and (d / "task.toml").is_file()
        )
    rows = []
    ok = 0
    ok_status = {"already_present", "loaded_and_tagged", "pulled_and_tagged", "would_prepare"}
    for d in dirs:
        if not d.is_dir():
            rows.append({"task_id": d.name, "status": "missing_dir"})
            continue
        rec = prepare_one(
            d,
            runtime=runtime,
            mmap=mmap,
            tar_root=tar_root if tar_root is not None and tar_root.is_dir() else None,
            dry_run=dry_run,
            pull=pull,
            bake=bake,
        )
        rows.append(rec)
        if rec["status"] in ok_status and rec.get("tmux") != "bake_failed":
            ok += 1
    return {"n": len(rows), "ok": ok, "rows": rows}
