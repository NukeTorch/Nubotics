# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""nubotics serve | demo | gr00t-server"""
import argparse
import os
import sys
import time

HOME = os.path.expanduser(os.environ.get("NUKETORCH_HOME", "~/NukeTorch"))
ENGINE = os.path.join(HOME, "nuketorch-1.0.5-macos-arm64", "nuketorch-serve")
TEXT = os.path.join(HOME, "models", "Qwen3.6-35B-A3B-4bit-DFlash2")
MODEL = os.path.join(HOME, "models", "Nubotics")


def serve(argv):
    ap = argparse.ArgumentParser(prog="nubotics serve", description="Start the NukeTorch server with the Nubotics policy.")
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--model", default=MODEL, help="the Nubotics model folder")
    ap.add_argument("--text-model", default=TEXT, help="the text model bundle NukeTorch 1.0.5 loads beside the policy")
    ap.add_argument("--port", default="8080")
    a = ap.parse_args(argv)
    for path, what in ((a.engine, "NukeTorch"), (a.model, "the Nubotics model"), (a.text_model, "the text model")):
        if not os.path.exists(path):
            sys.exit(f"{what} is not at {path}; run install.sh")
    name = os.path.basename(a.text_model.rstrip("/")).replace("-DFlash2", "")
    cmd = [a.engine, "http",
           "--model", os.path.join(a.text_model, name),
           "--draft-model", os.path.join(a.text_model, name.replace("-4bit", "") + "-DFlash2"),
           "--policy-model", a.model, "--port", a.port]
    print(" ".join(cmd), flush=True)
    os.execv(a.engine, cmd)


def demo(argv):
    ap = argparse.ArgumentParser(prog="nubotics demo", description="One robot call with a generated observation.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--task", default="put both the alphabet soup and the tomato sauce in the basket")
    a = ap.parse_args(argv)
    import numpy as np
    from .policy import Policy
    policy = Policy(a.model, a.url)
    rng = np.random.default_rng(0)
    images = {c: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for c in policy.cameras}
    state = np.zeros(policy.state_dim, np.float32)
    policy.act(images, state, a.task)   # first call: warm-up
    t0 = time.perf_counter()
    actions = policy.act(images, state, a.task)
    ms = (time.perf_counter() - t0) * 1000
    np.set_printoptions(precision=4, suppress=True)
    print(f"task: {a.task}")
    print(f"actions: {actions.shape[0]} steps x {actions.shape[1]} values, call {ms:.2f} ms (client included)")
    print(actions[:4])
    policy.close()


def main():
    commands = {"serve": serve, "demo": demo}
    if len(sys.argv) < 2 or sys.argv[1] not in (*commands, "gr00t-server"):
        sys.exit(__doc__.strip())
    if sys.argv[1] == "gr00t-server":
        from .gr00t_server import main as gr00t
        return gr00t(sys.argv[2:])
    return commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
