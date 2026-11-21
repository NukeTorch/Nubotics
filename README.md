# Nubotics 1.7B on NukeTorch

**Nubotics 1.7B is our first robotics model built for Mac.**
- A speculative eye model.
- Two-exponent 8-bit activations.
- A one-step action flow.
- 16+ independent robots on a single Mac.
- 18.57× faster than GR00T N1.7 on the same Mac while using 85% less memory and 89% less energy\*.
- On a MacBook Pro M5 Max, 1.22× faster than the GR00T N1.7 figure on NVIDIA's H100 (18.81 ms against 22.9 ms a call)\*\*.

For now Nubotics runs NVIDIA's GR00T N1.7 weights (fine-tuned on LIBERO-10); next, we train its weights ourselves.

\* GR00T N1.7 on its maker's Mac path: LeRobot 0.6.1 (Hugging Face) with NVIDIA's GR00T N1.7 checkpoint for LIBERO-10, as shipped (PyTorch 2.11.0 on the M5 Max GPU, 4 denoising steps), on the same MacBook Pro M5 Max: 349.3 ms a call, 27.1 GB peak RAM, 8.38 J for each action chunk. Nubotics: 18.81 ms, 4.1 GB, 0.90 J. Memory and energy on the development build of the same engine.

\*\* NVIDIA's published figure for GR00T N1.7 on LIBERO-10, H100 80GB HBM3 with its TensorRT pipeline, 4 denoising steps ([Isaac-GR00T deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)): 22.9 ms a call. Nubotics: NukeTorch 1.0.5's built-in benchmark on a MacBook Pro M5 Max with 48 GB, one robot, one call at a time, median, 18.81 ms.

**[Video demo](https://youtu.be/tctNRxkyYt8)** · **[Detailed benchmark report](benchmark/nubotics_M5.pdf)**

Simulation (LIBERO-10): [one robot](https://youtu.be/tctNRxkyYt8) · [8 robots, 8 different instructions in one call](https://youtu.be/mPabfuphjLg)

## Architecture

One call takes two camera frames, the arm's state and an instruction, and returns a chunk of 16 actions.

**Speculative eye model.** A fast, trained eye does the seeing: it reads the two cameras and the instruction and hands the action model what the full vision-language model would. It was distilled from the full model's own outputs on LIBERO-10 training episodes and checked against the full model before it shipped.

**Two-exponent int8.** All linear layers but one compute in 8-bit, weights and activations alike (the action model's entry stays 16-bit), and each value carries two scales: one for its column, set from the weights and from activations calibrated once on recorded frames, and one for its row, set from the row's largest value at run time. Two scales keep the 8-bit products close to 16-bit, so the whole model can run in 8-bit.

**One-step flow.** The policy's flow is solved in one denoising step instead of four.

**Many robots in one call.** The server gathers the waiting requests of separate robots, each with its own instruction, into one call. A laboratory or a production line can run 16 robots, each on its own task, from a single MacBook Pro.

**Weights.** Nubotics runs NVIDIA's GR00T N1.7 weights today; NVIDIA's license applies to them. Next: we train the weights ourselves.

## Benchmark

One MacBook Pro M5 Max with 48 GB, on AC power, after a rest, macOS 27.2.1.

**One robot, one call at a time**

| | Nubotics, NukeTorch | GR00T N1.7, maker's Mac path¹ | NVIDIA H100, TensorRT² |
|---|---:|---:|---:|
| One call (median) | **18.81 ms** | 349.3 ms | 22.9 ms |
| Action chunks a second | **53.2** | 2.86 | 43.7 |
| RAM, peak³ | **4.1 GB** | 27.1 GB | N/A |
| Power, CPU + GPU³ | 48.8 W | 24.0 W | N/A |
| Energy for each action chunk³ | **0.90 J** | 8.38 J | N/A |
| LIBERO-10 task success, 500 episodes³ | **89.2%** (446 of 500) | N/A | N/A |

Against NVIDIA's own H100 figure with TensorRT, Nubotics answers 1.22× faster (18.81 ms against 22.9 ms). Against GR00T N1.7 on its maker's Mac path, Nubotics is 18.57× faster, uses 85% less memory and 89% less energy for each action chunk.

**Several robots on one Mac.** Each robot sends its own observation; the robots waiting together are answered in one call.

| Robots at once | Action chunks a second, in total | Each robot's answer within |
|---:|---:|---:|
| 2 | **67.2** | 29.8 ms |
| 4 | **79.3** | 50.5 ms |
| 8 | **86.2** | 92.8 ms |
| 16 | **86.6** | 181.7 ms |

**Sustained: speed, power and RAM³.** Recorded camera streams of LIBERO-10 runs, a fresh chunk for every frame, back to back, one policy process.

| Robots | Action chunks a second | Call, median | Power, CPU + GPU | Chunks a joule | RAM, peak |
|---|---:|---:|---:|---:|---:|
| 1 robot | 54.2 | 18.4 ms | 48.8 W | 1.11 | 4.1 GB |
| 8 robots, one task | 83.5 | 95.8 ms | 63.0 W | 1.33 | 5.1 GB |
| 8 robots, 8 different tasks | 81.6 | 98.1 ms | 62.6 W | 1.30 | 5.0 GB |
| 16 robots, 8 different tasks (60 s) | 83.2 | 194.0 ms | 57.0 W | 1.46 | 8.0 GB |

**Real time, at the robot's own pace³.** Closed-loop LIBERO-10 episodes at 20 control steps a second, a new chunk every 8 steps, one process a robot.

| Robots | Calls inside the 50 ms step | Steps on time | Power, CPU + GPU | RAM, peak, all |
|---|---:|---:|---:|---:|
| 1 robot | 74.4% | 88.0% | 4.9 W | 6.3 GB |
| 1 robot, camera read every step | 100.0% | 98.2% | 12.3 W | 6.2 GB |
| 8 robots | 95.9% | 97.5% | 33.8 W | 44.8 GB |

At the robot's own pace the GPU waits between chunks and macOS lowers its clock; reading the camera on every step keeps the clock up.

**Have an M5 Ultra?** If your Mac Studio with an M5 Ultra has arrived, we'd love to hear how Nubotics runs on it. Run `nuketorch-serve bench` on the model and send us the report: contact@nuketorch.ai.

¹ LeRobot 0.6.1 (Hugging Face) with NVIDIA's GR00T N1.7 checkpoint for LIBERO-10, as shipped: PyTorch 2.11.0 on the M5 Max GPU (MPS), 4 denoising steps; 10 warm-up and 20 timed calls in 3 processes, median of the 60 calls, on this Mac.

² NVIDIA's published figure for GR00T N1.7 on LIBERO-10, H100 with its TensorRT pipeline, 4 denoising steps ([Isaac-GR00T deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)).

³ Measured on the development build of the same engine (M5 Max, macOS 27.2.1): task success, RAM, power and energy, the 2, 4 and 8 robots rows, the sustained and real-time tables. Power: CPU plus GPU from powermetrics. Every other Nubotics number on this page is NukeTorch 1.0.5's built-in benchmark (`nuketorch-serve bench`, model `nubotics-gr00t-n1.7`): one generated observation (two 256×256 camera frames, the arm's state and an instruction), 10 warm-up requests, then 256 one at a time; median; the best of 5 runs.

## Install

Use `curl`, not a browser.

```sh
curl -fsSL https://github.com/NukeTorch/nuketorch/releases/download/1.0.5/install.sh | sh
```

The script downloads the package, checks its SHA-256, shows the license and asks you to accept it, then unpacks it into `~/NukeTorch`. Nothing is installed outside that folder.

Requirements: a Mac with an M5 Max chip and macOS 27.2 or later. We tested on 48 GB.

## Get the model

```sh
python3 -m venv ~/NukeTorch/hf-env && ~/NukeTorch/hf-env/bin/pip install -U huggingface_hub
~/NukeTorch/hf-env/bin/hf download NukeTorch/Nubotics --local-dir ~/NukeTorch/models/Nubotics
```

Model card: https://huggingface.co/NukeTorch/Nubotics

## Run

```sh
cd ~/NukeTorch/nuketorch-1.0.5-macos-arm64
./nuketorch-serve bench ~/NukeTorch/models/Nubotics \
  --text-model ~/NukeTorch/models/Qwen3.6-35B-A3B-4bit-DFlash2
./nuketorch-serve report
```

`report` sets your run beside ours and beside NVIDIA's H100 figure. Let the Mac rest a few minutes before you measure; a warm Mac is slower.

You are responsible for the safe use of this model with a real robot.

## Multi Machines: NukeTorch Enterprise

Our pipeline parallelism spreads one AI model across many machines to make a local AI data center. Through a single API, four MacBook Pros serve 192 independent robots at once.

NukeTorch Enterprise for Multi Machines is by invitation. Ask for an evaluation license:

[![Request a NukeTorch Enterprise evaluation license](https://img.shields.io/badge/NukeTorch%20Enterprise-Request%20an%20evaluation%20license-121F47?style=for-the-badge)](mailto:contact@nuketorch.ai?subject=NukeTorch%20Enterprise%20evaluation%20request&body=Name%3A%0AOrganization%3A%0AMacs%20%28how%20many%2C%20which%20chips%2C%20memory%29%3A%0AModel%20or%20workload%3A%0A)

## License

NukeTorch is free for personal use, schools and universities, and businesses and other organizations (not government entities or international organizations) that with their affiliates have less than US$10 million in revenue and fewer than 250 people, on up to 250 machines, each working alone. Everyone else may evaluate it for 30 days, then needs a commercial license: contact@nuketorch.ai. Terms are in [LICENSE.md](https://github.com/NukeTorch/nuketorch/blob/1.0.5/LICENSE.md).

Nubotics runs NVIDIA's GR00T N1.7 weights, licensed by NVIDIA Corporation under the NVIDIA Open Model License. Their vision-language backbone is NVIDIA Cosmos-Reason2, so the license asks us to say: Built on NVIDIA Cosmos.

**Powered by NukeTorch** · https://nuketorch.ai · Back to [NukeTorch/nuketorch](https://github.com/NukeTorch/nuketorch)
