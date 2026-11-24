# Nubotics 1.7B on NukeTorch

Nubotics is a robot policy for the Mac. One call takes two camera frames, the arm's state and an instruction, and returns a chunk of 16 actions.

Nubotics runs on NukeTorch; the client and integration are Apache 2.0.

**[Video demo](https://www.youtube.com/watch?v=tctNRxkyYt8)** · **[Benchmark report](benchmark/nubotics_M5.pdf)** · [8 robots, 8 different instructions in one call](https://www.youtube.com/watch?v=mPabfuphjLg)

## Results

These were measured on one MacBook Pro M5 Max with 48 GB, running macOS 27.2.1.

| | Nubotics on NukeTorch |
|---|---:|
| One robot call (median) | **18.81 ms** |
| Action chunks a second, one robot | **53.2** |
| RAM, peak | **4.1 GB** |
| Energy for each action chunk | **0.90 J** |
| LIBERO-10 task success, 500 episodes (simulation) | **89.2%** (446 of 500) |
| 16 robots at once | **86.6** action chunks a second in total, 181.7 ms a request |

| One call | Nubotics, M5 Max | GR00T N1.7, maker's Mac path, M5 Max¹ | NVIDIA H100, TensorRT² |
|---|---:|---:|---:|
| Time (median) | **18.81 ms** | 349.3 ms | 22.9 ms |
| Action chunks a second | **53.2** | 2.86 | 43.7 |
| RAM, peak | **4.1 GB** | 27.1 GB | |
| Energy for each action chunk | **0.90 J** | 8.38 J | |

On the same Mac, Nubotics is 18.57× faster than GR00T N1.7 on its maker's Mac path. It uses 85% less memory and 89% less energy for each action chunk. Its 18.81 ms is 1.22× faster than NVIDIA's published H100 figure.

The one-robot call and the 16-robot row come from NukeTorch 1.0.5's built-in benchmark, best of 5 runs. Task success, RAM and energy were measured on the development build of the same engine. The full method, the per-task results, and the 2-, 4- and 8-robot and real-time runs are in the [report](benchmark/nubotics_M5.pdf).

¹ LeRobot 0.6.1 (Hugging Face) with NVIDIA's GR00T N1.7 checkpoint for LIBERO-10, as shipped: PyTorch 2.11.0 on the M5 Max GPU (MPS), 4 denoising steps.
² NVIDIA's published figure for GR00T N1.7 on LIBERO-10, on an H100 with its TensorRT pipeline and 4 denoising steps ([Isaac-GR00T deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)).

## Quickstart

You need a Mac with an M5 Max and macOS 27.2 or later. The installer downloads about 37 GB: NukeTorch, the Nubotics model, and the text model that NukeTorch 1.0.5 serves beside the policy.

```sh
curl -fsSL https://github.com/NukeTorch/Nubotics/raw/main/install.sh | sh
~/NukeTorch/nubotics-env/bin/nubotics serve        # starts NukeTorch with Nubotics; wait for READY
```

In a second terminal:

```sh
~/NukeTorch/nubotics-env/bin/nubotics demo         # one robot call: prints the 16 actions
```

NukeTorch's installer shows the NukeTorch license and asks you to accept it. Everything installs into `~/NukeTorch`.

## Use it from Python

```python
from nubotics import Policy

policy = Policy()                         # the model in ~/NukeTorch/models/Nubotics, the server from `nubotics serve`
actions = policy.act({"image": front_rgb, "wrist_image": wrist_rgb},   # uint8 [H][W][3]
                     state,                                            # float32 [8], robot units
                     "put the bowl on the plate")                      # -> float32 [16][7], robot units
```

`policy.act_many([...])` sends several robots at once. The server answers the robots that are waiting together in one call, and each robot can have its own instruction.

## Connect an existing robot stack

| You have | Use |
|---|---|
| NVIDIA Isaac-GR00T `PolicyClient` | `nubotics gr00t-server --port 5555`. The protocol, messages and keys are the same, so point your client at the Mac. |
| LeRobot | `pip install ./integrations/lerobot`, then `--policy.type=nubotics` |
| Your own loop | [`examples/mujoco_loop.py`](examples/mujoco_loop.py): LIBERO-10 in the MuJoCo simulator, closed loop at 20 Hz |
| Any language | `POST /v1/nuketorch/policy/act` on the NukeTorch server ([`src/nubotics/client.py`](src/nubotics/client.py) documents the request) |

## How it works

- **Speculative eye model.** A fast, trained eye reads the two cameras and the instruction. It hands the action model what the full vision-language model would. It was distilled from the full model's own outputs on LIBERO-10 training episodes.
- **Two-scale int8.** All linear layers but one compute in 8-bit, for both weights and activations.
- **One-step flow.** The policy's flow is solved in one denoising step instead of four.
- **Many robots in one call.** The server gathers the waiting requests of separate robots, each with its own instruction, into one call.

This repository holds the client side:
- the checkpoint's own input and output processing
- the HTTP client
- the GR00T and LeRobot integrations
- the installer

The model runs inside NukeTorch.

```
src/nubotics/          client: processor.py, client.py, policy.py, formats.py, gr00t_server.py, cli.py, embodiments/
integrations/lerobot/  LeRobot policy type "nubotics"
examples/              mujoco_loop.py
tests/                 client tests (no server needed)
install.sh             NukeTorch, client, models
benchmark/             nubotics_M5.pdf
```

## Tests

```sh
~/NukeTorch/nubotics-env/bin/python -m unittest discover tests
```

These tests run on any Mac and cover the request format, the GR00T and LeRobot formats, and the checkpoint's processing. `nubotics demo` and `examples/mujoco_loop.py` call the model, so they need the M5 Max.

You are responsible for the safe use of this model with a real robot.

## License

The client and integration in this repository are Apache 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

NukeTorch is distributed separately under its own license: [LICENSE.md](https://github.com/NukeTorch/nuketorch/blob/1.0.5/LICENSE.md).

Nubotics runs NVIDIA's GR00T N1.7 weights. They are licensed by NVIDIA Corporation under the NVIDIA Open Model License, and the model files carry that license and its notice ([model card](https://huggingface.co/NukeTorch/Nubotics)). Built on NVIDIA Cosmos.

**Powered by NukeTorch** · https://nuketorch.ai · contact@nuketorch.ai
