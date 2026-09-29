import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from multi_sample_inference.r3_checkout_binding import validate_checkout_route_binding
from multi_sample_inference.r3_pristine_adapter import (
    install_pristine_runtime_observer,
)
from multi_sample_inference.r3_route_isolation import (
    run_isolated_module,
    validate_exact_route_binding,
    validate_pristine_route_contract,
)

ROOT = Path(__file__).resolve().parents[1]
ROUTE_CONTRACT = ROOT / "docs/r3_pristine_route.json"


@pytest.fixture
def worker_module(monkeypatch):
    wan = types.ModuleType("wan")
    wan.__path__ = []
    configs = types.ModuleType("wan.configs")
    configs.__path__ = []
    config = types.ModuleType("wan.configs.wan_i2v_14B")
    config.i2v_14B = object()
    for name, module in {
        "wan": wan,
        "wan.configs": configs,
        "wan.configs.wan_i2v_14B": config,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = importlib.import_module("multi_sample_inference.fsdp_worker")
    yield module
    sys.modules.pop("multi_sample_inference.fsdp_worker", None)


class FakeTensor:
    def __init__(self, label):
        self.label = label

    def squeeze(self, _dimension):
        return self


class FakeScheduler:
    def step(self, *_args, **_kwargs):
        return [FakeTensor("scheduled")]


class FakeSelfAttention:
    def forward(self, value, *_args, **_kwargs):
        return sys.modules["wan.modules.model"].flash_attention(value, value, value)


class FakeCrossAttention:
    def forward(self, value, *_args, **_kwargs):
        flash = sys.modules["wan.modules.model"].flash_attention
        flash(value, value, value)
        return flash(value, value, value)


class FakeBlock:
    def __init__(self):
        self.self_attn = FakeSelfAttention()
        self.cross_attn = FakeCrossAttention()

    def forward(self, value, *_args, **_kwargs):
        value = self.self_attn.forward(value)
        return self.cross_attn.forward(value)


class FakeModel:
    def __init__(self):
        self.blocks = [FakeBlock()]

    def forward(self, values, *_args, **_kwargs):
        value = values[0]
        for block in self.blocks:
            value = block.forward(value)
        return [FakeTensor("prediction")]


class FakeGenerator:
    def __init__(self):
        self.model = FakeModel()

    def generate(self, steps):
        scheduler = FakeScheduler()
        latent = FakeTensor("initial")
        for _ in range(steps):
            self.model.forward([latent])
            self.model.forward([latent])
            latent = scheduler.step()[0]
        return latent


@pytest.fixture
def pristine_modules(monkeypatch):
    wan = types.ModuleType("wan")
    wan.__path__ = []
    image = types.ModuleType("wan.image2video")
    model = types.ModuleType("wan.modules.model")
    attention = types.ModuleType("wan.modules.attention")

    def fake_flash(q, _k, _v, version=None):
        del version
        return q

    package = types.SimpleNamespace(__version__="2.8.3")
    attention.FLASH_ATTN_2_AVAILABLE = True
    attention.FLASH_ATTN_3_AVAILABLE = False
    attention.flash_attn = package
    attention.flash_attn_interface = types.SimpleNamespace(__version__="3.0.0")
    model.flash_attention = fake_flash
    image.FlowUniPCMultistepScheduler = FakeScheduler
    image.FlowDPMSolverMultistepScheduler = type(
        "FakeDPMScheduler", (FakeScheduler,), {}
    )
    classes = (
        FakeScheduler,
        FakeSelfAttention,
        FakeCrossAttention,
        FakeBlock,
        FakeModel,
    )
    original_modules = {cls: cls.__module__ for cls in classes}
    original_names = {FakeModel: FakeModel.__name__, FakeGenerator: FakeGenerator.__name__}
    for cls in classes:
        cls.__module__ = "wan.modules.model"
    FakeModel.__name__ = "WanModel"
    FakeGenerator.__module__ = "wan.image2video"
    FakeGenerator.__name__ = "WanI2V"
    for name, module in {
        "wan": wan,
        "wan.image2video": image,
        "wan.modules.model": model,
        "wan.modules.attention": attention,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        yield image, model
    finally:
        for cls, module_name in original_modules.items():
            cls.__module__ = module_name
        FakeGenerator.__module__ = __name__
        for cls, name in original_names.items():
            cls.__name__ = name


def test_pristine_route_contract_binds_exact_cpu_implementation():
    contract = json.loads(ROUTE_CONTRACT.read_text())
    frozen = contract["implementation"]["worker_integration"]["sha256"]
    live = hashlib.sha256((ROOT / "multi_sample_inference/fsdp_worker.py").read_bytes()).hexdigest()
    if live == frozen:
        assert validate_pristine_route_contract(contract, ROOT)
    else:
        with pytest.raises(ValueError, match="worker_integration source changed"):
            validate_pristine_route_contract(contract, ROOT)


def test_pristine_adapter_observes_exact_coordinates_and_restores_wrappers(
    pristine_modules,
):
    image, model_module = pristine_modules
    generator = FakeGenerator()
    events = []
    originals = {
        "model": FakeModel.forward,
        "block": FakeBlock.forward,
        "self": FakeSelfAttention.forward,
        "cross": FakeCrossAttention.forward,
        "scheduler": FakeScheduler.step,
        "flash": model_module.flash_attention,
    }

    with install_pristine_runtime_observer(
        generator,
        events.append,
        rank=0,
        diffusion_seed=123,
        sampling_steps=2,
        expected_backend="flash_attention_2",
        expected_backend_version="2.8.3",
    ):
        generator.generate(2)

    assert [event["event"] for event in events[:2]] == [
        "observer-installed",
        "initial-latent",
    ]
    assert events[-1]["event"] == "observer-completed"
    assert len([event for event in events if event["event"] == "cfg-branch-output"]) == 4
    dispatches = [event for event in events if event["event"] == "attention-dispatch"]
    assert len(dispatches) == 12
    assert {
        (event["step"], event["branch"], event["block"], event["attention_site"])
        for event in dispatches
    } == {
        (step, branch, 0, site)
        for step in range(2)
        for branch in ("conditional", "negative")
        for site in ("self", "cross")
    }
    final = [event for event in events if event["event"] == "final-latent"]
    assert len(final) == 1
    assert final[0]["output_tensor"].label == "scheduled"
    assert FakeModel.forward is originals["model"]
    assert FakeBlock.forward is originals["block"]
    assert FakeSelfAttention.forward is originals["self"]
    assert FakeCrossAttention.forward is originals["cross"]
    assert FakeScheduler.step is originals["scheduler"]
    assert model_module.flash_attention is originals["flash"]
    assert image.FlowUniPCMultistepScheduler is FakeScheduler


def test_pristine_adapter_fails_closed_and_restores_after_incomplete_generation(
    pristine_modules,
):
    _, model_module = pristine_modules
    generator = FakeGenerator()
    events = []
    original_model = FakeModel.forward
    original_flash = model_module.flash_attention

    with pytest.raises(
        ValueError, match="cardinality is incomplete"
    ), install_pristine_runtime_observer(
        generator,
        events.append,
        rank=0,
        diffusion_seed=123,
        sampling_steps=2,
        expected_backend="flash_attention_2",
        expected_backend_version="2.8.3",
    ):
        generator.generate(1)

    assert events[-1] == {
        "event": "observer-failed",
        "rank": 0,
        "error_type": "ValueError",
    }
    assert FakeModel.forward is original_model
    assert model_module.flash_attention is original_flash


def _fake_wan_root(tmp_path, name, marker):
    root = tmp_path / name
    package = root / "wan"
    package.mkdir(parents=True)
    source = f"R3_TEST_MARKER = {marker!r}\n"
    (package / "__init__.py").write_text(source)
    return root, hashlib.sha256(source.encode()).hexdigest()


def test_pristine_and_custom_routes_use_distinct_clean_python_processes(tmp_path):
    pristine, pristine_hash = _fake_wan_root(tmp_path, "pristine", "official")
    custom, custom_hash = _fake_wan_root(tmp_path, "custom", "local")
    adapter = ROOT / "multi_sample_inference/r3_pristine_adapter.py"
    assert validate_exact_route_binding(
        {
            "route": "official-pristine",
            "wan_root": str(pristine),
            "source_files": {"wan/__init__.py": pristine_hash},
            "adapter": {
                "path": str(adapter),
                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            },
        }
    )
    assert validate_exact_route_binding(
        {
            "route": "local-custom",
            "wan_root": str(custom),
            "source_files": {"wan/__init__.py": custom_hash},
            "adapter": None,
        }
    )

    records = []
    for route, wan_root, marker in (
        ("official-pristine", pristine, "official"),
        ("local-custom", custom, "local"),
    ):
        output = tmp_path / f"{route}.json"
        run_isolated_module(
            python=sys.executable,
            route=route,
            wan_root=wan_root,
            project_root=ROOT,
            target_module="multi_sample_inference.r3_route_isolation",
            args=(
                "--probe",
                "--route",
                route,
                "--wan-root",
                wan_root,
                "--output",
                output,
                "--expected-marker",
                marker,
            ),
        )
        records.append(json.loads(output.read_text()))

    assert records[0]["pid"] != records[1]["pid"]
    assert [record["marker"] for record in records] == ["official", "local"]
    assert all(record["gpu_or_model_used"] is False for record in records)
    for record in records:
        assert set(record["wan_modules"]) == {"wan"}
        assert all(
            Path(origin).is_relative_to(Path(record["wan_root"]))
            for origin in record["wan_modules"]["wan"]
        )


def test_local_observer_scopes_only_pre_model_uncoordinated_dispatches(worker_module):
    records = []
    observer = worker_module._scope_local_dispatch_observer(records.append)
    observer({"event": "observer-installed", "rank": 0})
    observer({"event": "attention-dispatch", "rank": 0})  # Not after initial latent.
    observer({"event": "initial-latent", "rank": 0, "seed": 101})
    observer({"event": "attention-dispatch", "rank": 0, "backend": "flash_attention_2"})
    observer({"event": "attention-dispatch", "rank": 0, "step": 0})  # Partial scope must fail validation.
    observer({"event": "attention-dispatch", "rank": 0, "backend": "flash_attention_2"})
    assert [item["event"] for item in records] == [
        "observer-installed", "attention-dispatch", "initial-latent",
        "pre-model-attention-dispatch", "attention-dispatch", "attention-dispatch",
    ]
    assert records[3]["scope"] == "pre-model" and records[3]["rank"] == 0
    assert records[2]["seed"] == 101


def test_parity_custom_none_retains_fp32_weights_and_bf16_runtime(monkeypatch, worker_module):
    original_config = types.SimpleNamespace(param_dtype=torch.bfloat16)
    monkeypatch.setattr(worker_module, "i2v_14B", original_config)
    captured = []

    class FakeWanI2V:
        def __init__(self, *, config, **kwargs):
            captured.append((config, kwargs))
            self.config = config
            self.param_dtype = config.param_dtype
            self.model = torch.nn.Linear(2, 2).to(config.param_dtype)

    regional = types.ModuleType("wan.regional_prompt")
    regional.__path__ = []
    image2video = types.ModuleType("wan.regional_prompt.image2video")
    image2video.WanI2V = FakeWanI2V
    monkeypatch.setitem(sys.modules, "wan.regional_prompt", regional)
    monkeypatch.setitem(sys.modules, "wan.regional_prompt.image2video", image2video)
    task = {
        "config": {"bias_method": "none"},
        "checkpoint_dir": "/unused",
        "r3_evidence": {
            "protocol_schema_version": 12,
            "route_process": {"route": "local-custom"},
            "parity_artifact": {"route": "custom-none"},
        },
    }
    kwargs = {"local_rank": 0, "rank": 0, "t5_fsdp": False,
              "dit_fsdp": False, "t5_cpu": True}
    matched = worker_module._build_generator(task, **kwargs)
    assert matched.config is not original_config
    assert captured[0][0].param_dtype == torch.float32
    assert next(matched.model.parameters()).dtype == torch.float32
    assert matched.param_dtype == original_config.param_dtype == torch.bfloat16
    assert captured[0][1]["init_on_cpu"] is True

    ordinary = worker_module._build_generator({**task, "r3_evidence": {}}, **kwargs)
    assert captured[1][0] is original_config
    assert next(ordinary.model.parameters()).dtype == torch.bfloat16
    legacy = worker_module._build_generator(
        {**task, "r3_evidence": {**task["r3_evidence"], "protocol_schema_version": 11}},
        **kwargs,
    )
    assert next(legacy.model.parameters()).dtype == torch.bfloat16

    class RoundedWanI2V(FakeWanI2V):
        def __init__(self, *, config, **kwargs):
            super().__init__(config=config, **kwargs)
            self.model.to(torch.bfloat16)

    monkeypatch.setattr(image2video, "WanI2V", RoundedWanI2V)
    with pytest.raises(ValueError, match="did not retain checkpoint FP32 dtype"):
        worker_module._build_generator(task, **kwargs)
    with pytest.raises(ValueError, match="single-rank local custom-none"):
        worker_module._build_generator(task, **(kwargs | {"dit_fsdp": True}))
    with pytest.raises(ValueError, match="single-rank local custom-none"):
        worker_module._build_generator(
            {**task, "config": {"bias_method": "regional_prompting"}}, **kwargs
        )
    with pytest.raises(ValueError, match="single-rank local custom-none"):
        worker_module._build_generator(
            {**task, "r3_evidence": {**task["r3_evidence"],
                                     "route_process": {"route": "official-pristine"}}}, **kwargs
        )


def test_worker_selects_external_adapter_only_for_schema9_pristine_route(
    monkeypatch, worker_module
):
    adapter = importlib.import_module("multi_sample_inference.r3_pristine_adapter")
    sentinel = object()
    captured = {}

    def install(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(adapter, "install_pristine_runtime_observer", install)
    task = {
        "config": {"bias_method": "upstream"},
        "prompt_sentences": ["joint prompt"],
        "prompt_representation": "joint",
        "negative_prompt": "negative",
        "r3_evidence": {
            "protocol_schema_version": 9,
            "expected": {"sampling_steps": 2},
            "requested": {
                "diffusion_seed": 123,
                "attention_backend": "flash_attention_2",
                "dispatch_contract": {
                    "backend_versions": {"flash_attention_2": "2.8.3"}
                },
            },
        },
    }
    generator, collector = object(), object()

    assert worker_module._runtime_observer_context(
        generator, task, collector, 0
    ) is sentinel
    assert captured == {
        "args": (generator, collector),
        "kwargs": {
            "rank": 0,
            "diffusion_seed": 123,
            "sampling_steps": 2,
            "expected_backend": "flash_attention_2",
            "expected_backend_version": "2.8.3",
        },
    }


def test_worker_validates_schema9_route_before_model_loading(monkeypatch, worker_module):
    isolation = importlib.import_module("multi_sample_inference.r3_route_isolation")
    checkout_binding = importlib.import_module("multi_sample_inference.r3_checkout_binding")
    calls = []
    monkeypatch.setattr(
        checkout_binding,
        "validate_checkout_route_binding",
        lambda binding: calls.append(("binding", binding)),
    )
    monkeypatch.setattr(
        isolation,
        "assert_loaded_wan_modules",
        lambda root: calls.append(("root", root)),
    )
    binding = {"route": "official-pristine", "wan_root": "/pristine"}
    task = {
        "config": {"bias_method": "upstream"},
        "prompt_sentences": ["joint prompt"],
        "prompt_representation": "joint",
        "negative_prompt": "negative",
        "r3_evidence": {"protocol_schema_version": 9, "route_process": binding},
    }

    worker_module._validate_r3_route_process_before_model_load(task)
    assert calls == [("binding", binding), ("root", "/pristine")]
    binding["route"] = "local-custom"
    with pytest.raises(ValueError, match="worker route differs"):
        worker_module._validate_r3_route_process_before_model_load(task)


def test_whole_checkout_binding_rejects_drift_and_requires_detached_pristine(tmp_path):
    pristine, file_hash = _fake_wan_root(tmp_path, "pristine", "official")
    subprocess.run(["git", "-C", str(pristine), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(pristine), "add", "wan/__init__.py"], check=True)
    subprocess.run(
        ["git", "-C", str(pristine), "-c", "user.name=R3 Test", "-c",
         "user.email=r3@example.invalid", "commit", "-qm", "fixture"], check=True,
    )
    adapter = tmp_path / "adapter.py"
    adapter.write_text("# observer\n")
    binding = {
        "route": "official-pristine", "wan_root": str(pristine),
        "source_files": {"wan/__init__.py": file_hash},
        "adapter": {"path": str(adapter), "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()},
        "checkout": {
            "commit": subprocess.check_output(["git", "-C", str(pristine), "rev-parse", "HEAD"], text=True).strip(),
            "tree": subprocess.check_output(["git", "-C", str(pristine), "rev-parse", "HEAD^{tree}"], text=True).strip(),
        },
    }
    with pytest.raises(ValueError, match="detached HEAD"):
        validate_checkout_route_binding(binding)
    subprocess.run(["git", "-C", str(pristine), "checkout", "--detach", "-q"], check=True)
    assert validate_checkout_route_binding(binding)
    binding["checkout"]["tree"] = "0" * 40
    with pytest.raises(ValueError, match="commit or tree differs"):
        validate_checkout_route_binding(binding)
    binding["checkout"]["tree"] = subprocess.check_output(
        ["git", "-C", str(pristine), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    (pristine / "other.py").write_text("# untracked import surface\n")
    with pytest.raises(ValueError, match="untracked"):
        validate_checkout_route_binding(binding)
    (pristine / "other.py").unlink()
    (pristine / ".git/info/exclude").write_text("wan/ignored.pyc\n")
    (pristine / "wan/ignored.pyc").write_bytes(b"untrusted cache")
    with pytest.raises(ValueError, match="ignored importable"):
        validate_checkout_route_binding(binding)


def test_exact_route_binding_rejects_source_or_adapter_tampering(tmp_path):
    pristine, pristine_hash = _fake_wan_root(tmp_path, "pristine", "official")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("before\n")
    binding = {
        "route": "official-pristine",
        "wan_root": str(pristine),
        "source_files": {"wan/__init__.py": pristine_hash},
        "adapter": {
            "path": str(adapter),
            "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        },
    }
    adapter.write_text("after\n")
    with pytest.raises(ValueError, match="adapter changed"):
        validate_exact_route_binding(binding)

    binding["adapter"]["sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    (pristine / "wan/__init__.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        validate_exact_route_binding(binding)
