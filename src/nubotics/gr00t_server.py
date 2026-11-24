# Copyright 2026 NukeTorch
# SPDX-License-Identifier: Apache-2.0
"""A policy server that answers NVIDIA Isaac-GR00T's PolicyClient unchanged, so a robot or simulation that talks to a GR00T server
today can point at the Mac instead: same host:port, same messages, same keys.

Protocol (gr00t/policy/server_client.py, commit 3b49067943eb1ab2eb2203b93d461c4390c78da3): ZeroMQ REQ/REP over TCP, one msgpack
message each way, numpy arrays in msgpack_numpy's form. Request {"endpoint", "data", "api_token"}; endpoints:
  ping -> {"status": "ok", "message": "Server is running"}     get_action {observation, options} -> [actions, info]
  reset {options} -> {}     get_modality_config -> {modality: ModalityConfig}     kill -> stops the server
A failure travels back as {"error": text}. Both observation forms are taken (nested and flat); the answer comes back in the form the
question had. Actions are un-normalized and the gripper is left as the model gives it, as NVIDIA's server returns it.
Every client waiting when the policy is free goes into one batched call.

    nubotics gr00t-server --port 5555 [--host ADDRESS] [--api-token T]

It listens on this Mac only unless --host says otherwise: the protocol has no authentication beyond the optional token, and anyone who
can reach the port can stop the server (the kill endpoint).
"""
import argparse
import io
import json
import sys
import traceback

import numpy as np

from .client import LOCAL
from .formats import Embodiment, from_gr00t, gr00t_modality_config, to_gr00t


def _wire():
    try:
        import msgpack
        import msgpack_numpy as mnp
        import zmq
    except ImportError:
        sys.exit("the GR00T server needs: pip install 'nubotics[gr00t]'")

    def encode(obj):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(f"Refusing to encode object-dtype ndarray (shape={obj.shape})")
        return mnp.encode(obj)

    def decode(obj):
        if isinstance(obj, dict):
            if obj.get("__ndarray_class__", obj.get(b"__ndarray_class__")):   # Isaac-GR00T's older array form
                payload = obj.get("as_npy", obj.get(b"as_npy"))
                if payload is None:
                    raise ValueError("Malformed ndarray payload: marker present but 'as_npy' missing")
                return np.load(io.BytesIO(payload), allow_pickle=False)
            if obj.get(b"nd", obj.get("nd")) and obj.get(b"kind", obj.get("kind")) in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload")
        return mnp.decode(obj)

    pack = lambda data: msgpack.packb(data, default=encode)
    unpack = lambda data: msgpack.unpackb(data, object_hook=decode, raw=False)
    return zmq, pack, unpack


class Server:
    def __init__(self, policy, embodiment, host=LOCAL, port=5555, api_token=None, max_batch=16):
        self.zmq, self.pack, self.unpack = _wire()
        self.policy, self.emb, self.api_token, self.max_batch = policy, embodiment, api_token, max_batch
        self.context = self.zmq.Context()
        self.socket = self.context.socket(self.zmq.ROUTER)   # a REQ client cannot tell it from REP; it lets waiting clients share a call
        self.socket.bind(f"tcp://{host}:{port}")
        self.address = self.socket.getsockopt_string(self.zmq.LAST_ENDPOINT)
        self.running = True

    def close(self):
        self.socket.close(linger=1000)
        self.context.term()

    def _take(self, block):
        try:
            frames = self.socket.recv_multipart(0 if block else self.zmq.NOBLOCK)
        except self.zmq.Again:
            return None
        envelope, payload = frames[:-1], frames[-1]
        try:
            request = self.unpack(payload)
            if not isinstance(request, dict):
                raise ValueError("The request must be a dictionary")
        except Exception as e:
            return envelope, {"error": str(e)}, None
        if self.api_token is not None and request.get("api_token") != self.api_token:
            return envelope, {"error": "Unauthorized: Invalid API token"}, None
        return envelope, None, request

    def _reply(self, envelope, result):
        try:
            payload = self.pack(result)
        except Exception as e:
            payload = self.pack({"error": str(e)})
        self.socket.send_multipart(envelope + [payload])

    def run(self):
        print(f"Server is ready and listening on {self.address}", flush=True)
        while self.running:
            waiting = [self._take(True)]
            while len(waiting) < self.max_batch:
                more = self._take(False)
                if more is None:
                    break
                waiting.append(more)
            asks = []   # (envelope, rows, flat)
            for envelope, refusal, request in waiting:
                if refusal is not None:
                    self._reply(envelope, refusal)
                    continue
                try:
                    endpoint = request.get("endpoint", "get_action")
                    data = request.get("data") or {}
                    if endpoint == "ping":
                        self._reply(envelope, {"status": "ok", "message": "Server is running"})
                    elif endpoint == "kill":
                        self.running = False
                        self._reply(envelope, None)
                    elif endpoint == "reset":
                        self._reply(envelope, {})
                    elif endpoint == "get_modality_config":
                        self._reply(envelope, gr00t_modality_config(self.emb))
                    elif endpoint == "get_action":
                        rows, flat = from_gr00t(data.get("observation"), self.emb)
                        asks.append((envelope, rows, flat))
                    else:
                        raise ValueError(f"Unknown endpoint: {endpoint}")
                except Exception as e:
                    print(f"Error in server: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                    self._reply(envelope, {"error": str(e)})
            if not asks:
                continue
            try:
                out = self.policy.act_many([o for _, rows, _ in asks for o in rows], steps=self.emb.action_horizon, env_gripper=False)
                at = 0
                for envelope, rows, flat in asks:
                    self._reply(envelope, [to_gr00t(out[at:at + len(rows)], self.emb, flat), {}])
                    at += len(rows)
            except Exception as e:
                print(f"Error in server: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                for envelope, _, _ in asks:
                    self._reply(envelope, {"error": str(e)})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nubotics gr00t-server", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=LOCAL)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--api-token", default=None)
    ap.add_argument("--model", default=None, help="the Nubotics model folder (default ~/NukeTorch/models/Nubotics)")
    ap.add_argument("--url", default=None, help="the NukeTorch server (default: the one nubotics serve starts)")
    ap.add_argument("--embodiment", default=None, help="an embodiment tag in nubotics/embodiments or a JSON file; default: the checkpoint's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=16)
    a = ap.parse_args(argv)
    from .policy import Policy
    policy = Policy(a.model, a.url, seed=a.seed, max_batch=a.max_batch)
    emb = Embodiment.load(a.embodiment or policy.embodiment_tag)
    if emb.video != policy.cameras or emb.state_dim != policy.state_dim or emb.action_dim != policy.action_dim:
        sys.exit(f"embodiment {emb.tag} does not match the checkpoint: cameras {emb.video} / {policy.cameras}, "
                 f"state {emb.state_dim} / {policy.state_dim}, action {emb.action_dim} / {policy.action_dim}")
    server = Server(policy, emb, a.host, a.port, a.api_token, a.max_batch)
    try:
        server.run()
    finally:
        server.close()
        policy.close()


if __name__ == "__main__":
    main()
