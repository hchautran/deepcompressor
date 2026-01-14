# -*- coding: utf-8 -*-
"""SAM2 Hiera model structure for quantization."""

import typing as tp
from collections import OrderedDict
from dataclasses import dataclass, field

import torch.nn as nn

from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

__all__ = ["SAM2ModelStruct", "SAM2BlockStruct", "SAM2AttentionStruct", "SAM2MLPStruct"]


@dataclass(kw_only=True)
class SAM2ModuleStruct(BaseModuleStruct):
    """Base structure for SAM2 modules."""

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        if isinstance(self.module, nn.Linear):
            yield self.key, self.name, self.module, self.parent, self.fname
        else:
            for name, module in self.module.named_modules():
                if name and isinstance(module, nn.Linear):
                    module_name = join_name(self.name, name)
                    field_name = join_name(self.fname, name)
                    yield self.key, module_name, module, self.parent, field_name


@dataclass(kw_only=True)
class SAM2AttentionStruct(SAM2ModuleStruct):
    """SAM2 MultiScaleAttention structure.

    In SAM2 Hiera, attention consists of:
    - qkv: Fused Q, K, V projection (nn.Linear)
    - proj: Output projection (nn.Linear)
    """

    qkv: nn.Linear
    proj: nn.Linear
    qkv_name: str = field(init=False)
    proj_name: str = field(init=False)

    def __post_init__(self):
        self.qkv_name = join_name(self.name, "qkv")
        self.proj_name = join_name(self.name, "proj")

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        yield "attn.qkv", self.qkv_name, self.qkv, self, "qkv"
        yield "attn.proj", self.proj_name, self.proj, self, "proj"

    @staticmethod
    def construct(
        module: nn.Module,
        /,
        parent: tp.Optional["BaseModuleStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "SAM2AttentionStruct":
        name = join_name(rname, "attn")
        key = join_name(rkey, "attn")
        return SAM2AttentionStruct(
            module=module,
            parent=parent,
            name=name,
            key=key,
            fname=join_name(fname, "attn"),
            idx=idx,
            qkv=module.qkv,
            proj=module.proj,
        )


@dataclass(kw_only=True)
class SAM2MLPStruct(SAM2ModuleStruct):
    """SAM2 MLP structure.

    In SAM2 Hiera, MLP consists of:
    - layers.0: First linear layer (up projection)
    - layers.1: Second linear layer (down projection)
    """

    fc1: nn.Linear
    fc2: nn.Linear
    fc1_name: str = field(init=False)
    fc2_name: str = field(init=False)

    def __post_init__(self):
        self.fc1_name = join_name(self.name, "layers.0")
        self.fc2_name = join_name(self.name, "layers.1")

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        yield "mlp.fc1", self.fc1_name, self.fc1, self, "fc1"
        yield "mlp.fc2", self.fc2_name, self.fc2, self, "fc2"

    @staticmethod
    def construct(
        module: nn.Module,
        /,
        parent: tp.Optional["BaseModuleStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "SAM2MLPStruct":
        name = join_name(rname, "mlp")
        key = join_name(rkey, "mlp")
        return SAM2MLPStruct(
            module=module,
            parent=parent,
            name=name,
            key=key,
            fname=join_name(fname, "mlp"),
            idx=idx,
            fc1=module.layers[0],
            fc2=module.layers[1],
        )


@dataclass(kw_only=True)
class SAM2BlockStruct(SAM2ModuleStruct):
    """SAM2 MultiScaleBlock structure.

    Each block contains:
    - norm1: LayerNorm before attention
    - attn: MultiScaleAttention
    - norm2: LayerNorm before MLP
    - mlp: MLP
    """

    attn_struct: SAM2AttentionStruct
    mlp_struct: SAM2MLPStruct

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        yield from self.attn_struct.named_key_modules()
        yield from self.mlp_struct.named_key_modules()

    def iter_attention_structs(self) -> tp.Generator[SAM2AttentionStruct, None, None]:
        yield self.attn_struct

    def iter_mlp_structs(self) -> tp.Generator[SAM2MLPStruct, None, None]:
        yield self.mlp_struct

    @staticmethod
    def construct(
        module: nn.Module,
        /,
        parent: tp.Optional["BaseModuleStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "SAM2BlockStruct":
        name = join_name(rname, f"blocks.{idx}")
        key = join_name(rkey, "block")
        fname = join_name(fname, f"blocks.{idx}")

        attn_struct = SAM2AttentionStruct.construct(
            module.attn,
            parent=None,
            fname=fname,
            rname=name,
            rkey=key,
            idx=idx,
        )

        mlp_struct = SAM2MLPStruct.construct(
            module.mlp,
            parent=None,
            fname=fname,
            rname=name,
            rkey=key,
            idx=idx,
        )

        block_struct = SAM2BlockStruct(
            module=module,
            parent=parent,
            name=name,
            key=key,
            fname=fname,
            idx=idx,
            attn_struct=attn_struct,
            mlp_struct=mlp_struct,
        )
        attn_struct.parent = block_struct
        mlp_struct.parent = block_struct

        return block_struct


@dataclass(kw_only=True)
class SAM2ModelStruct(SAM2ModuleStruct):
    """SAM2 Hiera backbone model structure.

    The Hiera backbone consists of:
    - patch_embed: Patch embedding layer
    - blocks: List of MultiScaleBlocks
    """

    block_structs: list[SAM2BlockStruct] = field(init=False, repr=False)
    pre_module_structs: OrderedDict[str, SAM2ModuleStruct] = field(init=False, repr=False)
    post_module_structs: OrderedDict[str, SAM2ModuleStruct] = field(init=False, repr=False)

    @property
    def num_blocks(self) -> int:
        return len(self.block_structs)

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        for module in self.pre_module_structs.values():
            yield from module.named_key_modules()
        for block in self.block_structs:
            yield from block.named_key_modules()
        for module in self.post_module_structs.values():
            yield from module.named_key_modules()

    def iter_attention_structs(self) -> tp.Generator[SAM2AttentionStruct, None, None]:
        for block in self.block_structs:
            yield from block.iter_attention_structs()

    def iter_mlp_structs(self) -> tp.Generator[SAM2MLPStruct, None, None]:
        for block in self.block_structs:
            yield from block.iter_mlp_structs()

    def iter_block_structs(self) -> tp.Generator[SAM2BlockStruct, None, None]:
        yield from self.block_structs

    def get_iter_layer_activations_args(
        self, skip_pre_modules: bool = True, skip_post_modules: bool = True, **input_kwargs
    ) -> tuple[list[nn.Module], list[SAM2BlockStruct], list[bool], list[bool]]:
        """Get arguments for iterating over block activations."""
        layers = [block.module for block in self.block_structs]
        structs = list(self.block_structs)
        recomputes = [False] * len(layers)
        use_prev_layer_outputs = [True] * len(layers)
        use_prev_layer_outputs[0] = False
        return layers, structs, recomputes, use_prev_layer_outputs

    @staticmethod
    def construct(
        module: nn.Module,
        /,
        parent: tp.Optional["BaseModuleStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "SAM2ModelStruct":
        """Construct SAM2ModelStruct from a SAM2 model.

        Args:
            module: SAM2 model or SAM2 image/video predictor

        Returns:
            SAM2ModelStruct for the Hiera backbone
        """
        # Handle different SAM2 model wrappers
        if hasattr(module, "model"):
            # SAM2VideoPredictor or SAM2ImagePredictor
            module = module.model

        if hasattr(module, "image_encoder"):
            # SAM2Base model
            image_encoder = module.image_encoder
            if hasattr(image_encoder, "trunk"):
                # ImageEncoder wrapper
                backbone = image_encoder.trunk
            else:
                backbone = image_encoder
        elif hasattr(module, "trunk"):
            backbone = module.trunk
        elif hasattr(module, "blocks"):
            # Direct Hiera backbone
            backbone = module
        else:
            raise ValueError(f"Cannot find Hiera backbone in module: {type(module)}")

        name = rname or "backbone"
        key = rkey or "backbone"

        model_struct = SAM2ModelStruct(
            module=backbone,
            parent=parent,
            name=name,
            key=key,
            fname=fname or name,
            idx=idx,
        )

        # Build block structures
        block_structs = []
        for i, block in enumerate(backbone.blocks):
            block_struct = SAM2BlockStruct.construct(
                block,
                parent=model_struct,
                fname=name,
                rname=name,
                rkey=key,
                idx=i,
            )
            block_structs.append(block_struct)

        model_struct.block_structs = block_structs
        model_struct.pre_module_structs = OrderedDict()
        model_struct.post_module_structs = OrderedDict()

        return model_struct

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default key map for SAM2 modules."""
        key_map: dict[str, set[str]] = {
            "attn": set(),
            "attn.qkv": set(),
            "attn.proj": set(),
            "mlp": set(),
            "mlp.fc1": set(),
            "mlp.fc2": set(),
            "block": set(),
        }
        # These will be populated dynamically based on the actual model
        return key_map
