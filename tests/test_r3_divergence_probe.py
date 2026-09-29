import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from multi_sample_inference.r3_divergence_probe import (
    MANIFEST_NAME,
    _Capture,
    _write_manifest,
    build_launch_command,
    compare_captures,
    install_probe,
)

ROOT = Path(__file__).resolve().parents[1]
NEGATIVE = "blurred details"


def _encoder_types():
    class FakeT5:
        def __call__(self, texts, device):
            return [torch.full((len(text) % 5 + 2, 4), float(sum(map(ord, text)) % 97))
                    for text in texts]

    class FakeCLIP:
        def visual(self, videos):
            return torch.stack([video.flatten()[:8] for video in videos])

    class FakeVAE:
        def encode(self, videos):
            return [video * 2 for video in videos]

    return FakeT5, FakeCLIP, FakeVAE


class Block(nn.Module):
    def __init__(self, dim, *, tuple_output, fail=False):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.tuple_output = tuple_output
        self.fail = fail

    def forward(self, x, context):
        if self.fail:
            raise ValueError("synthetic block failure")
        out = x + self.linear(x) + context.mean()
        return (out, None, None) if self.tuple_output else out


def _modules(model, *, tuple_output, fail_block=None):
    torch.manual_seed(0)
    model.patch_embedding = nn.Linear(4, 4)
    model.time_embedding = nn.Linear(1, 4)
    model.time_projection = nn.Linear(4, 4)
    model.text_embedding = nn.Linear(4, 4)
    model.img_emb = nn.Linear(8, 4)
    model.blocks = nn.ModuleList(
        Block(4, tuple_output=tuple_output, fail=index == fail_block) for index in range(3)
    )
    model.head = nn.Linear(4, 4)
    model.freqs = torch.polar(torch.ones(4, dtype=torch.float64), torch.arange(4.0, dtype=torch.float64))


def _body(model, x, t, context, clip_fea, y, *, extra_text=False):
    hidden = model.patch_embedding(x[0] + y[0])
    e0 = model.time_projection(model.time_embedding(t.float().reshape(1, 1)))
    text = model.text_embedding(context[0])
    full = torch.cat([model.img_emb(clip_fea), text])
    if extra_text:  # custom-style later text_embedding call must not be captured
        model.text_embedding(context[0] * 3)
    for block in model.blocks:
        out = block(hidden, full + e0.mean())
        hidden = out[0] if isinstance(out, tuple) else out
    return model.head(hidden)


class PristineModel(nn.Module):
    def __init__(self, fail_block=None):
        super().__init__()
        _modules(self, tuple_output=False, fail_block=fail_block)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        return [_body(self, x, t, context, clip_fea, y)]


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        _modules(self, tuple_output=True)

    def forward(self, x, t, context, seq_len, bias_kwargs, clip_fea, y):
        assert bias_kwargs["bias_method"] == "none"
        return [_body(self, x, t, context, clip_fea, y, extra_text=True)], None, None


class Generator:
    def __init__(self, model, encoders, *, custom, prompt="a cat", negative=True, steps=3):
        self.model = model
        self.t5, self.clip, self.vae = (kind() for kind in encoders)
        self.custom = custom
        self.prompt = prompt
        self.negative = negative
        self.steps = steps
        self.calls = 0

    def _model(self, **kwargs):
        self.calls += 1
        if self.custom:
            return self.model(self.latent, **kwargs, bias_kwargs={"bias_method": "none"})[0][0]
        return self.model(self.latent, **kwargs)[0]

    def generate(self):
        if self.custom:  # the custom route encodes the negative prompt first
            [context_null] = self.t5([NEGATIVE], "cpu")
            [context] = self.t5([self.prompt], "cpu")
        else:
            [context] = self.t5([self.prompt], "cpu")
            [context_null] = self.t5([NEGATIVE], "cpu")
        image = torch.linspace(-1, 1, 20).reshape(5, 4)
        clip_fea = self.clip.visual([image])
        [y] = self.vae.encode([image])
        self.latent = [torch.ones(5, 4)]
        for step in range(self.steps):
            shared = {"t": torch.tensor([1000 - step]), "clip_fea": clip_fea, "seq_len": 5, "y": [y]}
            self._model(context=[context], **shared)
            if self.negative:
                self._model(context=[context_null], **shared)
        return "video"


def _manifest(directory):
    return json.loads((directory / MANIFEST_NAME).read_text())


def _no_hooks(model):
    return all(not module._forward_hooks for module in model.modules())


def test_capture_stops_after_step_zero_negative_forward_and_restores(tmp_path):
    encoders = _encoder_types()
    originals = [encoders[0].__call__, encoders[1].visual, encoders[2].encode,
                 PristineModel.forward]
    generator = Generator(PristineModel(), encoders, custom=False)
    with install_probe(generator, tmp_path / "capture", encoder_types=encoders):
        generator.generate()

    assert generator.calls == 2
    manifest = _manifest(tmp_path / "capture")
    assert manifest["status"] == "complete" and manifest["r3_acceptance"] is False
    names = [record["name"] for record in manifest["records"]]
    assert len([name for name in names if name.startswith("encoder/t5/")]) == 2
    for expected in ("encoder/clip/0/input/0", "encoder/clip/0/output/0",
                     "encoder/vae/0/input/0", "encoder/vae/0/output/0",
                     "dit/conditional/input/x/0", "dit/conditional/input/seq_len",
                     "dit/conditional/freqs", "dit/conditional/patch_embedding",
                     "dit/conditional/img_emb", "dit/conditional/block/02",
                     "dit/conditional/head", "dit/conditional/output",
                     "dit/negative/input/context/0", "dit/negative/output"):
        assert expected in names
    assert not any(name.startswith("dit/negative/block") for name in names)
    assert [encoders[0].__call__, encoders[1].visual, encoders[2].encode,
            PristineModel.forward] == originals
    assert _no_hooks(generator.model)
    seq_len = next(r for r in manifest["records"] if r["name"] == "dit/conditional/input/seq_len")
    assert seq_len["value"] == 5


def test_generation_without_stop_fails_closed(tmp_path):
    encoders = _encoder_types()
    generator = Generator(PristineModel(), encoders, custom=False, negative=False, steps=1)
    with pytest.raises(RuntimeError, match="without reaching"), install_probe(
        generator, tmp_path / "capture", encoder_types=encoders
    ):
        generator.generate()
    assert _manifest(tmp_path / "capture")["status"] == "failed"
    assert PristineModel.forward.__name__ == "forward" and _no_hooks(generator.model)


def test_forward_failure_restores_wrappers_and_records_failure(tmp_path):
    encoders = _encoder_types()
    original = PristineModel.forward
    generator = Generator(PristineModel(fail_block=1), encoders, custom=False)
    with pytest.raises(ValueError, match="synthetic block"), install_probe(
        generator, tmp_path / "capture", encoder_types=encoders
    ):
        generator.generate()
    manifest = _manifest(tmp_path / "capture")
    assert manifest["status"] == "failed" and "synthetic block" in manifest["error"]
    assert PristineModel.forward is original and _no_hooks(generator.model)


def _capture(tmp_path, name, model, *, custom, prompt="a cat", replay=None):
    encoders = _encoder_types()
    generator = Generator(model, encoders, custom=custom, prompt=prompt)
    with install_probe(generator, tmp_path / name, encoder_types=encoders, replay_dir=replay):
        generator.generate()
    return tmp_path / name


def test_comparator_localizes_first_divergent_block(tmp_path):
    reference = _capture(tmp_path, "pristine", PristineModel(), custom=False)
    candidate_model = CustomModel()
    with torch.no_grad():
        candidate_model.blocks[1].linear.bias.add_(1e-3)
    candidate = _capture(tmp_path, "local", candidate_model, custom=True)

    report = compare_captures(reference, candidate)
    assert report["r3_acceptance"] is False
    # T5 captures are keyed by prompt text, so differing call order still matches.
    assert report["first_divergent_encoder_stage"] is None
    assert report["first_divergent_dit_stage"] == "dit/conditional/block/01"
    stages = {stage["name"]: stage for stage in report["stages"]}
    assert stages["dit/conditional/block/00"]["status"] == "equal"
    assert stages["dit/conditional/text_embedding"]["status"] == "equal"
    metrics = stages["dit/conditional/block/01"]["metrics"]
    assert metrics["max_abs_error"] == pytest.approx(1e-3, rel=0.1)
    assert report["candidate_only_stages"] == []


def test_replay_separates_conditioning_from_dit_arithmetic(tmp_path):
    reference = _capture(tmp_path, "pristine", PristineModel(), custom=False)
    native = _capture(tmp_path, "local", CustomModel(), custom=True, prompt="a dog")
    replay = _capture(tmp_path, "replay", CustomModel(), custom=True, prompt="a dog",
                      replay=reference)

    native_report = compare_captures(reference, native)
    assert native_report["first_divergent_encoder_stage"].startswith("encoder/t5/")
    assert native_report["first_divergent_dit_stage"] == "dit/conditional/input/context/0"

    replay_report = compare_captures(reference, replay)
    assert replay_report["first_divergent_dit_stage"] is None
    assert replay_report["candidate"]["replay"]["path"] == str(reference.resolve())
    assert len(replay_report["candidate"]["replay"]["manifest_sha256"]) == 64


def test_replay_requires_complete_reference(tmp_path):
    encoders = _encoder_types()
    failed = tmp_path / "failed"
    generator = Generator(PristineModel(fail_block=0), encoders, custom=False)
    with pytest.raises(ValueError), install_probe(generator, failed, encoder_types=encoders):
        generator.generate()
    candidate = Generator(CustomModel(), encoders, custom=True)
    with pytest.raises(ValueError, match="incomplete"), install_probe(
        candidate, tmp_path / "replay", encoder_types=encoders, replay_dir=failed
    ):
        candidate.generate()
    assert CustomModel.forward.__name__ == "forward"


def test_dtype_mismatch_reports_upcast_metrics(tmp_path):
    for name, dtype in (("reference", torch.float32), ("candidate", torch.bfloat16)):
        capture = _Capture(tmp_path / name)
        capture.save("dit/conditional/head", torch.tensor([1.0, 2.5, -3.0], dtype=dtype))
        _write_manifest(capture, {"route": name}, "complete")
    report = compare_captures(tmp_path / "reference", tmp_path / "candidate")
    [stage] = report["stages"]
    assert stage["status"] == "different"
    assert stage["metrics"]["dtype_match"] is False
    assert stage["metrics"]["upcast"]["max_abs_error"] == 0.0
    assert report["first_divergent_dit_stage"] == "dit/conditional/head"


def test_launch_command_uses_isolated_single_rank_route(tmp_path):
    wan_root = tmp_path / "wan"
    wan_root.mkdir()
    command = build_launch_command(
        python=sys.executable, route="local-custom", wan_root=wan_root, project_root=ROOT,
        task_file=tmp_path / "task.pkl", output=tmp_path / "out", replay=tmp_path / "ref",
    )
    assert command[1] == "-I"
    assert json.loads(command[4]) == [str(wan_root.resolve()), str(ROOT), "torch.distributed.run"]
    assert command[5:9] == ["--nproc_per_node=1", "-m",
                            "multi_sample_inference.r3_divergence_probe", "capture"]
    assert command[-2:] == ["--replay", str((tmp_path / "ref").resolve())]
    with pytest.raises(ValueError, match="replay requires local-custom"):
        build_launch_command(
            python=sys.executable, route="official-pristine", wan_root=wan_root,
            project_root=ROOT, task_file=tmp_path / "task.pkl", output=tmp_path / "out",
            replay=tmp_path / "ref",
        )
