"""External runtime observations for the immutable pristine Wan I2V route.

This module is deliberately outside the pristine checkout. It observes the
official generator by temporarily wrapping Python call boundaries and restores
every wrapped attribute on exit; it never edits vendor source files.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager

_INSTALLED = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _first_output(value, label):
    _require(isinstance(value, (list, tuple)) and value, f"{label} output is malformed")
    return value[0]


def _actual_flash_backend(attention_module, version):
    fa2 = attention_module.FLASH_ATTN_2_AVAILABLE
    fa3 = attention_module.FLASH_ATTN_3_AVAILABLE
    _require(type(fa2) is bool and type(fa3) is bool, "pristine attention availability is malformed")
    if (version is None or version == 3) and fa3:
        package = attention_module.flash_attn_interface
        return "flash_attention_3", getattr(package, "__version__", None)
    _require(fa2, "pristine FlashAttention 2 is unavailable")
    package = attention_module.flash_attn
    return "flash_attention_2", getattr(package, "__version__", None)


def _argument(function, args, kwargs, name, default=None):
    try:
        index = function.__code__.co_varnames.index(name)
    except (AttributeError, ValueError):
        return kwargs.get(name, default)
    if name in kwargs:
        return kwargs[name]
    return args[index] if index < len(args) else default


class _PristineInstrumentation:
    def __init__(
        self,
        generator,
        observer,
        *,
        rank,
        diffusion_seed,
        sampling_steps,
        expected_backend,
        expected_backend_version,
    ):
        _require(callable(observer), "pristine runtime observer must be callable")
        _require(type(rank) is int and rank >= 0, "pristine runtime rank is invalid")
        _require(type(diffusion_seed) is int, "pristine diffusion seed is invalid")
        _require(type(sampling_steps) is int and sampling_steps > 0, "pristine sampling steps are invalid")
        _require(expected_backend == "flash_attention_2", "pristine route requires FlashAttention 2")
        _require(
            isinstance(expected_backend_version, str) and expected_backend_version,
            "pristine FlashAttention version is invalid",
        )
        _require(
            (type(generator).__module__, type(generator).__name__)
            == ("wan.image2video", "WanI2V"),
            "generator is not pristine wan.image2video.WanI2V",
        )
        self.generator = generator
        self.observer = observer
        self.rank = rank
        self.diffusion_seed = diffusion_seed
        self.sampling_steps = sampling_steps
        self.expected_backend = expected_backend
        self.expected_backend_version = expected_backend_version
        self.image_module = importlib.import_module("wan.image2video")
        self.model_module = importlib.import_module("wan.modules.model")
        self.attention_module = importlib.import_module("wan.modules.attention")
        self.model = generator.model
        _require(
            (type(self.model).__module__, type(self.model).__name__)
            == ("wan.modules.model", "WanModel"),
            "pristine generator model type is unexpected",
        )
        self.blocks = list(getattr(self.model, "blocks", ()))
        _require(self.blocks, "pristine generator has no attention blocks")
        self.block_indices = {id(block): index for index, block in enumerate(self.blocks)}
        self.attention_sites = {
            id(attention): (index, site)
            for index, block in enumerate(self.blocks)
            for site, attention in (("self", block.self_attn), ("cross", block.cross_attn))
        }
        self.context = {}
        self.counts = {"model": 0, "scheduler": 0, "initial": 0, "final": 0}
        self.restorations = []

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self.restorations.append((owner, name, original))
        return original

    def emit(self, event, **values):
        self.observer({"event": event, "rank": self.rank, **self.context, **values})

    def install_model(self):
        model_type = type(self.model)
        original = model_type.forward

        def model_forward(current_model, *args, **kwargs):
            if current_model is not self.model:
                return original(current_model, *args, **kwargs)
            call = self.counts["model"]
            _require(call < 2 * self.sampling_steps, "pristine model call cardinality exceeded")
            latent_inputs = args[0] if args else kwargs.get("x")
            _require(
                isinstance(latent_inputs, (list, tuple)) and len(latent_inputs) == 1,
                "pristine model latent input is malformed",
            )
            if call == 0:
                self.emit("initial-latent", seed=self.diffusion_seed, latent_tensor=latent_inputs[0])
                self.counts["initial"] += 1
            previous = dict(self.context)
            self.context.update(
                {
                    "step": call // 2,
                    "branch": "conditional" if call % 2 == 0 else "negative",
                }
            )
            try:
                result = original(current_model, *args, **kwargs)
                self.emit("cfg-branch-output", output_tensor=_first_output(result, "pristine model"))
            finally:
                self.context.clear()
                self.context.update(previous)
            self.counts["model"] += 1
            return result

        self.patch(model_type, "forward", model_forward)

    def install_blocks(self):
        for block_type in {type(block) for block in self.blocks}:
            original = block_type.forward

            def block_forward(block, *args, __original=original, **kwargs):
                index = self.block_indices.get(id(block))
                if index is None:
                    return __original(block, *args, **kwargs)
                _require("block" not in self.context, "pristine block observation context overlaps")
                self.context["block"] = index
                try:
                    return __original(block, *args, **kwargs)
                finally:
                    self.context.pop("block")

            self.patch(block_type, "forward", block_forward)

    def install_attention(self):
        attention_types = {type(block.self_attn) for block in self.blocks} | {
            type(block.cross_attn) for block in self.blocks
        }
        for attention_type in attention_types:
            original = attention_type.forward

            def attention_forward(attention, *args, __original=original, **kwargs):
                coordinate = self.attention_sites.get(id(attention))
                if coordinate is None:
                    return __original(attention, *args, **kwargs)
                block, site = coordinate
                _require(self.context.get("block") == block, "pristine attention block context differs")
                _require("attention_site" not in self.context, "pristine attention context overlaps")
                self.context["attention_site"] = site
                try:
                    return __original(attention, *args, **kwargs)
                finally:
                    self.context.pop("attention_site")

            self.patch(attention_type, "forward", attention_forward)

    def install_flash(self):
        original = self.model_module.flash_attention

        def observed_flash(*args, **kwargs):
            _require(
                set(self.context) >= {"step", "branch", "block", "attention_site"},
                "pristine attention dispatch lacks runtime coordinates",
            )
            version = _argument(original, args, kwargs, "version")
            backend, backend_version = _actual_flash_backend(self.attention_module, version)
            _require(backend == self.expected_backend, "pristine attention backend differs from its binding")
            _require(
                backend_version == self.expected_backend_version,
                "pristine attention backend version differs from its binding",
            )
            result = original(*args, **kwargs)
            self.emit("attention-dispatch", backend=backend, backend_version=backend_version)
            return result

        self.patch(self.model_module, "flash_attention", observed_flash)

    def install_schedulers(self):
        scheduler_types = {
            self.image_module.FlowUniPCMultistepScheduler,
            self.image_module.FlowDPMSolverMultistepScheduler,
        }
        for scheduler_type in scheduler_types:
            original = scheduler_type.step

            def scheduler_step(scheduler, *args, __original=original, **kwargs):
                result = __original(scheduler, *args, **kwargs)
                self.counts["scheduler"] += 1
                _require(
                    self.counts["scheduler"] <= self.sampling_steps,
                    "pristine scheduler step cardinality exceeded",
                )
                if self.counts["scheduler"] == self.sampling_steps:
                    final = _first_output(result, "pristine scheduler").squeeze(0)
                    self.emit("final-latent", output_tensor=final)
                    self.counts["final"] += 1
                return result

            self.patch(scheduler_type, "step", scheduler_step)

    def install(self):
        self.install_model()
        self.install_blocks()
        self.install_attention()
        self.install_flash()
        self.install_schedulers()

    def validate_complete(self):
        expected = {
            "model": 2 * self.sampling_steps,
            "scheduler": self.sampling_steps,
            "initial": 1,
            "final": 1,
        }
        _require(self.counts == expected, "pristine generator observation cardinality is incomplete")

    def restore(self):
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)


@contextmanager
def install_pristine_runtime_observer(
    generator,
    observer,
    *,
    rank,
    diffusion_seed,
    sampling_steps,
    expected_backend,
    expected_backend_version,
):
    """Observe one pinned official generation and fail closed on route drift."""
    global _INSTALLED
    _require(not _INSTALLED, "pristine runtime observation adapter is already installed")
    instrumentation = _PristineInstrumentation(
        generator,
        observer,
        rank=rank,
        diffusion_seed=diffusion_seed,
        sampling_steps=sampling_steps,
        expected_backend=expected_backend,
        expected_backend_version=expected_backend_version,
    )
    try:
        instrumentation.install()
        _INSTALLED = True
        instrumentation.emit("observer-installed")
    except Exception:
        instrumentation.restore()
        _INSTALLED = False
        raise
    try:
        yield observer
        instrumentation.validate_complete()
        instrumentation.emit("observer-completed")
    except Exception as error:
        instrumentation.emit("observer-failed", error_type=type(error).__name__)
        raise
    finally:
        instrumentation.restore()
        _INSTALLED = False
