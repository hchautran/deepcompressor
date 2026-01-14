# -*- coding: utf-8 -*-
"""SAM2 model struct helpers for quantization."""

from __future__ import annotations

import typing as tp
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import torch.nn as nn

from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

try:  # Optional import for type checks at runtime.
    from sam2.modeling.backbones.hieradet import Hiera, MultiScaleBlock
    from sam2.modeling.backbones.image_encoder import ImageEncoder
    from sam2.modeling.sam2_base import SAM2Base
except Exception:  # pragma: no cover - SAM2 may be unavailable in some envs.
    Hiera = MultiScaleBlock = ImageEncoder = SAM2Base = None

__all__ = ["Sam2ModelStruct", "Sam2BlockStruct", "Sam2ModuleStruct"]


@dataclass(kw_only=True)
class Sam2ModuleStruct(BaseModuleStruct):
    """Generic module struct for SAM2 components."""

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        if isinstance(self.module, (nn.Linear, nn.Conv2d)):
            yield self.key, self.name, self.module, self.parent, self.fname
            return
        for name, module in self.module.named_modules():
            if name and isinstance(module, (nn.Linear, nn.Conv2d)):
                module_name = join_name(self.name, name)
                module_key = join_name(self.key, name.replace(".", "_"), sep="_")
                yield module_key, module_name, module, self, name

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2ModuleStruct":
        return cls(module=module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)


@dataclass(kw_only=True)
class Sam2BlockStruct(BaseModuleStruct):
    """SAM2 Hiera block struct."""

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        attn = getattr(self.module, "attn", None)
        if attn is not None:
            qkv = getattr(attn, "qkv", None)
            if isinstance(qkv, nn.Linear):
                name = "attn.qkv"
                key = join_name(self.key, "attn_qkv", sep="_")
                yield key, join_name(self.name, name), qkv, self, "attn_qkv"
            proj = getattr(attn, "proj", None)
            if isinstance(proj, nn.Linear):
                name = "attn.proj"
                key = join_name(self.key, "attn_proj", sep="_")
                yield key, join_name(self.name, name), proj, self, "attn_proj"
        mlp = getattr(self.module, "mlp", None)
        if mlp is not None and hasattr(mlp, "layers"):
            for idx, layer in enumerate(mlp.layers):
                if isinstance(layer, nn.Linear):
                    name = f"mlp.layers.{idx}"
                    key = join_name(self.key, f"mlp_fc{idx + 1}", sep="_")
                    yield key, join_name(self.name, name), layer, self, f"mlp_fc{idx + 1}"
        proj = getattr(self.module, "proj", None)
        if isinstance(proj, nn.Linear):
            name = "proj"
            key = join_name(self.key, "proj", sep="_")
            yield key, join_name(self.name, name), proj, self, "proj"

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2BlockStruct":
        return cls(module=module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)


@dataclass(kw_only=True)
class Sam2ModelStruct(BaseModuleStruct):
    """SAM2 Hiera backbone struct."""

    pre_module_structs: OrderedDict[str, Sam2ModuleStruct] = field(init=False, repr=False)
    block_structs: list[Sam2BlockStruct] = field(init=False, repr=False)
    post_module_structs: OrderedDict[str, Sam2ModuleStruct] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        module = self.module
        if Hiera is not None and isinstance(module, Hiera):
            self.pre_module_structs = OrderedDict()
            self.block_structs = []
            self.post_module_structs = OrderedDict()
            # Patch embedding (Conv2d inside PatchEmbed)
            if hasattr(module, "patch_embed"):
                patch = Sam2ModuleStruct.construct(
                    module.patch_embed, parent=self, fname="pre_module_structs", rname="patch_embed", rkey="patch_embed"
                )
                self.pre_module_structs[patch.name] = patch
            # Hiera blocks
            for idx, block in enumerate(module.blocks):
                block_struct = Sam2BlockStruct.construct(
                    block,
                    parent=self,
                    fname="block_structs",
                    rname=f"blocks.{idx}",
                    rkey=f"blocks_{idx}",
                    idx=idx,
                )
                self.block_structs.append(block_struct)
            return
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

    @property
    def num_blocks(self) -> int:
        return len(self.block_structs)

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        for module in self.pre_module_structs.values():
            yield from module.named_key_modules()
        for block in self.block_structs:
            yield from block.named_key_modules()
        for module in self.post_module_structs.values():
            yield from module.named_key_modules()

    def get_named_layers(
        self, skip_pre_modules: bool, skip_post_modules: bool, skip_blocks: bool = False
    ) -> OrderedDict[str, Sam2BlockStruct | Sam2ModuleStruct]:
        named_layers: OrderedDict[str, Sam2BlockStruct | Sam2ModuleStruct] = OrderedDict()
        if not skip_pre_modules:
            named_layers.update(self.pre_module_structs)
        if not skip_blocks:
            for block in self.block_structs:
                named_layers[block.name] = block
        if not skip_post_modules:
            named_layers.update(self.post_module_structs)
        return named_layers

    def get_prev_module_keys(self) -> tuple[str, ...]:
        return tuple(module.key for module in self.pre_module_structs.values())

    def get_post_module_keys(self) -> tuple[str, ...]:
        return tuple(module.key for module in self.post_module_structs.values())

    def get_iter_layer_activations_args(
        self, skip_pre_modules: bool, skip_post_modules: bool, **input_kwargs
    ) -> tuple[list[nn.Module], list[Sam2BlockStruct | Sam2ModuleStruct], list[bool], list[bool]]:
        layers: list[nn.Module] = []
        layer_structs: list[Sam2BlockStruct | Sam2ModuleStruct] = []
        if not skip_pre_modules:
            for module in self.pre_module_structs.values():
                layers.append(module.module)
                layer_structs.append(module)
        for block in self.block_structs:
            layers.append(block.module)
            layer_structs.append(block)
        if not skip_post_modules:
            for module in self.post_module_structs.values():
                layers.append(module.module)
                layer_structs.append(module)
        recomputes = [False] * len(layers)
        use_prev_layer_outputs = [True] * len(layers)
        if use_prev_layer_outputs:
            use_prev_layer_outputs[0] = False
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        key_map: dict[str, set[str]] = defaultdict(set)
        # Keep each key as its own group to avoid strict simplification.
        for key in cls.get_default_keys():
            key_map[key].add(key)
        return dict(key_map)

    @staticmethod
    def _simplify_keys(keys: tp.Iterable[str], *, key_map: dict[str, set[str]]) -> list[str]:
        # For SAM2 we keep keys as-is to avoid unexpected assertion failures.
        return sorted(set(keys))

    @classmethod
    def get_default_keys(cls) -> list[str]:
        return [
            "patch_embed",
            "blocks",
            "attn_qkv",
            "attn_proj",
            "mlp_fc1",
            "mlp_fc2",
            "proj",
        ]

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2ModelStruct":
        if SAM2Base is not None and isinstance(module, SAM2Base):
            module = module.image_encoder.trunk
        elif ImageEncoder is not None and isinstance(module, ImageEncoder):
            module = module.trunk
        if Hiera is not None and isinstance(module, Hiera):
            return cls(module=module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)
        raise NotImplementedError(f"Unsupported module type: {type(module)}")
