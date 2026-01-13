# -*- coding: utf-8 -*-
"""SAM2 model structure definitions for quantization."""

import typing as tp
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import torch.nn as nn
from transformers.models.sam2.modeling_sam2 import (
    Sam2Model,
    Sam2VisionModel, 
    Sam2HieraDetModel,
    Sam2MultiScaleBlock,
    Sam2MultiScaleAttention,
    Sam2FeedForward
)

from deepcompressor.nn.struct.attn import (
    AttentionConfigStruct,
    AttentionStruct,
    FeedForwardConfigStruct,
    FeedForwardStruct,
    TransformerBlockStruct,
    
)
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

__all__ = [
    "Sam2ModelStruct",
    "Sam2VisionEncoderStruct",
    "Sam2HieraStruct",
    "Sam2HieraBlockStruct",
    "Sam2AttentionStruct",
    "Sam2FeedForwardStruct",
]


@dataclass(kw_only=True)
class Sam2ModuleStruct(BaseModuleStruct):
    """Base module struct for SAM2."""

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        if isinstance(self.module, nn.Linear):
            yield self.key, self.name, self.module, self.parent, self.fname
        else:
            for name, module in self.module.named_modules():
                if name and isinstance(module, nn.Linear):
                    module_name = join_name(self.name, name)
                    field_name = join_name(self.fname, name)
                    yield self.key, module_name, module, self.parent, field_name


@dataclass(kw_only=True)
class Sam2AttentionStruct(AttentionStruct):
    """SAM2 Hiera attention structure."""

    module: nn.Module = field(repr=False, kw_only=False)
    parent: tp.Optional["Sam2HieraBlockStruct"] = field(repr=False)

    def __post_init__(self) -> None:
        BaseModuleStruct.__post_init__(self)
        for field_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q",
            "k",
            "v",
        ):
            rname = getattr(self, f"{field_name}_rname")
            if getattr(self, field_name) is not None or rname:
                assert rname, f"`{field_name}_rname` must not be empty if `{field_name}` is not None"
                setattr(self, f"{field_name}_name", join_name(self.name, rname))
            else:
                setattr(self, f"{field_name}_name", "")
        self.qkv_proj_key = join_name(self.key, self.qkv_proj_rkey, sep="_")
        self.add_qkv_proj_key = join_name(self.key, self.add_qkv_proj_rkey, sep="_")
        self.out_proj_key = join_name(self.key, self.out_proj_rkey, sep="_")
        self.add_out_proj_key = join_name(self.key, self.add_out_proj_rkey, sep="_")
        self.q_key = join_name(self.key, self.q_rkey, sep="_")
        self.k_key = join_name(self.key, self.k_rkey, sep="_")
        self.v_key = join_name(self.key, self.v_rkey, sep="_")

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["Sam2HieraBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2AttentionStruct":
        """Construct Sam2AttentionStruct from a Hiera attention module."""
        # Hiera attention has qkv (fused) and proj
        qkv = getattr(module, "qkv", None)
        proj = getattr(module, "proj", None)

        # If qkv is fused, we need to handle it specially
        if qkv is not None:
            q_proj = qkv
            k_proj = None  # Will be handled as fused
            v_proj = None
            q_proj_rname = "qkv"
            k_proj_rname = ""
            v_proj_rname = ""
        else:
            # Separate q, k, v projections
            q_proj = getattr(module, "q_proj", getattr(module, "q", None))
            k_proj = getattr(module, "k_proj", getattr(module, "k", None))
            v_proj = getattr(module, "v_proj", getattr(module, "v", None))
            q_proj_rname = "q_proj" if hasattr(module, "q_proj") else "q"
            k_proj_rname = "k_proj" if hasattr(module, "k_proj") else "k"
            v_proj_rname = "v_proj" if hasattr(module, "v_proj") else "v"

        o_proj = proj
        o_proj_rname = "proj"

        # Get dimensions
        num_heads = getattr(module, "num_heads", 1)
        if qkv is not None:
            hidden_size = qkv.weight.shape[1]
            inner_size = qkv.weight.shape[0] // 3  # QKV fused
        else:
            hidden_size = q_proj.weight.shape[1]
            inner_size = q_proj.weight.shape[0]

        config = AttentionConfigStruct(
            hidden_size=hidden_size,
            add_hidden_size=0,
            inner_size=inner_size,
            num_query_heads=num_heads,
            num_key_value_heads=num_heads,
            with_qk_norm=False,
            with_rope=False,
            linear_attn=False,
        )

        return Sam2AttentionStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            add_q_proj=None,
            add_k_proj=None,
            add_v_proj=None,
            add_o_proj=None,
            q=None,
            k=None,
            v=None,
            q_proj_rname=q_proj_rname,
            k_proj_rname=k_proj_rname,
            v_proj_rname=v_proj_rname,
            o_proj_rname=o_proj_rname,
            add_q_proj_rname="",
            add_k_proj_rname="",
            add_v_proj_rname="",
            add_o_proj_rname="",
            q_rname="",
            k_rname="",
            v_rname="",
        )

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        """Yield quantizable modules."""
        # For fused QKV
        if self.k_proj is None and self.v_proj is None:
            yield self.qkv_proj_key, self.q_proj_name, self.q_proj, self, "q_proj"
        else:
            yield self.qkv_proj_key, self.q_proj_name, self.q_proj, self, "q_proj"
            if self.k_proj is not None:
                yield self.qkv_proj_key, self.k_proj_name, self.k_proj, self, "k_proj"
            if self.v_proj is not None:
                yield self.qkv_proj_key, self.v_proj_name, self.v_proj, self, "v_proj"
        yield self.out_proj_key, self.o_proj_name, self.o_proj, self, "o_proj"


@dataclass(kw_only=True)
class Sam2FeedForwardStruct(FeedForwardStruct):
    """SAM2 Hiera MLP/FFN structure."""

    module: nn.Module = field(repr=False, kw_only=False)
    parent: tp.Optional["Sam2HieraBlockStruct"] = field(repr=False)
    moe_gate: None = field(init=False, repr=False, default=None)
    experts: list[nn.Module] = field(init=False, repr=False)
    moe_gate_rname: str = field(init=False, repr=False, default="")
    experts_rname: str = field(init=False, repr=False, default="")

    @property
    def up_proj(self) -> nn.Linear:
        return self.up_projs[0]

    @property
    def down_proj(self) -> nn.Linear:
        return self.down_projs[0]

    @property
    def up_proj_rname(self) -> str:
        return self.up_proj_rnames[0]

    @property
    def down_proj_rname(self) -> str:
        return self.down_proj_rnames[0]

    @property
    def up_proj_name(self) -> str:
        return self.up_proj_names[0]

    @property
    def down_proj_name(self) -> str:
        return self.down_proj_names[0]

    def __post_init__(self) -> None:
        assert len(self.up_projs) == len(self.down_projs) == 1
        assert len(self.up_proj_rnames) == len(self.down_proj_rnames) == 1
        self.experts = [self.module]
        super().__post_init__()

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["Sam2HieraBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2FeedForwardStruct":
        """Construct Sam2FeedForwardStruct from a Hiera MLP module."""
        # Hiera MLP typically has fc1 (or lin1) and fc2 (or lin2)
        fc1 = getattr(module, "fc1", getattr(module, "lin1", None))
        fc2 = getattr(module, "fc2", getattr(module, "lin2", None))

        if fc1 is None or fc2 is None:
            # Try alternative names
            for name, submodule in module.named_modules():
                if isinstance(submodule, nn.Linear):
                    if fc1 is None:
                        fc1 = submodule
                        fc1_rname = name
                    elif fc2 is None:
                        fc2 = submodule
                        fc2_rname = name
        else:
            fc1_rname = "fc1" if hasattr(module, "fc1") else "lin1"
            fc2_rname = "fc2" if hasattr(module, "fc2") else "lin2"

        hidden_size = fc1.weight.shape[1]
        intermediate_size = fc1.weight.shape[0]

        config = FeedForwardConfigStruct(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            intermediate_act_type="gelu",
            num_experts=1,
        )

        return Sam2FeedForwardStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            up_projs=[fc1],
            down_projs=[fc2],
            up_proj_rnames=[fc1_rname],
            down_proj_rnames=[fc2_rname],
        )


@dataclass(kw_only=True)
class Sam2HieraBlockStruct(TransformerBlockStruct):
    """SAM2 Hiera transformer block structure."""

    norm_rkey: tp.ClassVar[str] = "hiera_norm"
    add_norm_rkey: tp.ClassVar[str] = ""
    attn_struct_cls: tp.ClassVar[type[Sam2AttentionStruct]] = Sam2AttentionStruct
    ffn_struct_cls: tp.ClassVar[type[Sam2FeedForwardStruct]] = Sam2FeedForwardStruct

    parent: tp.Optional["Sam2HieraStruct"] = field(repr=False)
    stage_idx: int = 0
    block_idx: int = 0

    # Child structs
    attn_structs: list[Sam2AttentionStruct] = field(init=False, repr=False)
    ffn_struct: Sam2FeedForwardStruct | None = field(init=False, repr=False)
    add_ffn_struct: None = field(init=False, repr=False, default=None)

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["Sam2HieraStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        stage_idx: int = 0,
        block_idx: int = 0,
        **kwargs,
    ) -> "Sam2HieraBlockStruct":
        """Construct Sam2HieraBlockStruct from a Hiera block module."""
        # Hiera block structure: norm1 -> attn -> norm2 -> mlp
        norm1 = getattr(module, "norm1", None)
        norm2 = getattr(module, "norm2", None)
        attn = getattr(module, "attn", None)
        mlp = getattr(module, "mlp", None)

        pre_attn_norms = [norm1] if norm1 is not None else []
        pre_attn_norm_rnames = ["norm1"] if norm1 is not None else []

        attns = [attn] if attn is not None else []
        attn_rnames = ["attn"] if attn is not None else []

        pre_ffn_norm = norm2
        pre_ffn_norm_rname = "norm2" if norm2 is not None else ""

        ffn = mlp
        ffn_rname = "mlp" if mlp is not None else ""

        return Sam2HieraBlockStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            stage_idx=stage_idx,
            block_idx=block_idx,
            parallel=False,
            pre_attn_norms=pre_attn_norms,
            pre_attn_add_norms=[],
            attns=attns,
            post_attn_norms=[],
            post_attn_add_norms=[],
            pre_ffn_norm=pre_ffn_norm,
            ffn=ffn,
            post_ffn_norm=None,
            pre_add_ffn_norm=None,
            add_ffn=None,
            post_add_ffn_norm=None,
            pre_attn_norm_rnames=pre_attn_norm_rnames,
            pre_attn_add_norm_rnames=[],
            attn_rnames=attn_rnames,
            post_attn_norm_rnames=[],
            post_attn_add_norm_rnames=[],
            pre_ffn_norm_rname=pre_ffn_norm_rname,
            ffn_rname=ffn_rname,
            post_ffn_norm_rname="",
            pre_add_ffn_norm_rname="",
            add_ffn_rname="",
            post_add_ffn_norm_rname="",
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default key map for quantization."""
        key_map: dict[str, set[str]] = defaultdict(set)
        attn_rkey = cls.attn_rkey
        qkv_proj_key = join_name(attn_rkey, cls.attn_struct_cls.qkv_proj_rkey, sep="_")
        out_proj_key = join_name(attn_rkey, cls.attn_struct_cls.out_proj_rkey, sep="_")
        key_map[attn_rkey].add(qkv_proj_key)
        key_map[attn_rkey].add(out_proj_key)
        key_map[cls.attn_struct_cls.qkv_proj_rkey].add(qkv_proj_key)
        key_map[cls.attn_struct_cls.out_proj_rkey].add(out_proj_key)

        ffn_rkey = cls.ffn_rkey
        up_proj_key = join_name(ffn_rkey, cls.ffn_struct_cls.up_proj_rkey, sep="_")
        down_proj_key = join_name(ffn_rkey, cls.ffn_struct_cls.down_proj_rkey, sep="_")
        key_map[ffn_rkey].add(up_proj_key)
        key_map[ffn_rkey].add(down_proj_key)
        key_map[cls.ffn_struct_cls.up_proj_rkey].add(up_proj_key)
        key_map[cls.ffn_struct_cls.down_proj_rkey].add(down_proj_key)

        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class Sam2HieraStruct(BaseModuleStruct):
    """SAM2 Hiera backbone structure."""

    module: nn.Module = field(repr=False, kw_only=False)
    parent: tp.Optional["Sam2VisionEncoderStruct"] = field(repr=False)

    # Patch embedding
    patch_embed: nn.Module | None = field(repr=False)
    patch_embed_rname: str

    # Stages and blocks
    blocks: nn.ModuleList = field(repr=False)
    blocks_rname: str
    blocks_per_stage: list[int]

    # Block structs
    block_structs: list[Sam2HieraBlockStruct] = field(init=False, repr=False)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Build block structs
        self.block_structs = []
        block_idx = 0
        for stage_idx, num_blocks in enumerate(self.blocks_per_stage):
            for i in range(num_blocks):
                block = self.blocks[block_idx]
                rname = f"{self.blocks_rname}.{block_idx}"
                block_struct = Sam2HieraBlockStruct.construct(
                    block,
                    parent=self,
                    fname="block",
                    rname=rname,
                    rkey="hiera_block",
                    idx=block_idx,
                    stage_idx=stage_idx,
                    block_idx=i,
                )
                self.block_structs.append(block_struct)
                block_idx += 1

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        for block in self.block_structs:
            yield from block.named_key_modules()

    def iter_transformer_block_structs(self) -> tp.Generator[Sam2HieraBlockStruct, None, None]:
        for block in self.block_structs:
            yield from block.iter_transformer_block_structs()

    def iter_attention_structs(self) -> tp.Generator[Sam2AttentionStruct, None, None]:
        for block in self.block_structs:
            yield from block.iter_attention_structs()

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["Sam2VisionEncoderStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2HieraStruct":
        """Construct Sam2HieraStruct from a Hiera backbone module."""
        # Get patch embedding
        patch_embed = getattr(module, "patch_embed", None)
        patch_embed_rname = "patch_embed" if patch_embed is not None else ""

        # Get blocks
        blocks = getattr(module, "blocks", None)
        blocks_rname = "blocks"

        # Get blocks_per_stage from config or infer
        config = getattr(module, "config", None)
        if config is not None:
            blocks_per_stage = list(config.blocks_per_stage)
        else:
            # Default for SAM2 Hiera-tiny: [1, 2, 7, 2]
            blocks_per_stage = [1, 2, 7, 2]

        return Sam2HieraStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            patch_embed=patch_embed,
            patch_embed_rname=patch_embed_rname,
            blocks=blocks,
            blocks_rname=blocks_rname,
            blocks_per_stage=blocks_per_stage,
        )


@dataclass(kw_only=True)
class Sam2VisionEncoderStruct(BaseModuleStruct):
    """SAM2 Vision Encoder structure."""

    module: nn.Module = field(repr=False, kw_only=False)
    parent: tp.Optional["Sam2ModelStruct"] = field(repr=False)

    # Backbone
    backbone: nn.Module = field(repr=False)
    backbone_rname: str

    # Neck/FPN
    neck: nn.Module | None = field(repr=False)
    neck_rname: str

    # Backbone struct
    backbone_struct: Sam2HieraStruct = field(init=False, repr=False)

    @property
    def num_blocks(self) -> int:
        return self.backbone_struct.num_blocks

    def __post_init__(self) -> None:
        super().__post_init__()
        self.backbone_struct = Sam2HieraStruct.construct(
            self.backbone,
            parent=self,
            fname="backbone",
            rname=self.backbone_rname,
            rkey="hiera",
        )

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        yield from self.backbone_struct.named_key_modules()

    def iter_transformer_block_structs(self) -> tp.Generator[Sam2HieraBlockStruct, None, None]:
        yield from self.backbone_struct.iter_transformer_block_structs()

    def iter_attention_structs(self) -> tp.Generator[Sam2AttentionStruct, None, None]:
        yield from self.backbone_struct.iter_attention_structs()

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: tp.Optional["Sam2ModelStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2VisionEncoderStruct":
        """Construct Sam2VisionEncoderStruct from a vision encoder module."""
        # Get backbone (Hiera)
        backbone = getattr(module, "backbone", None)
        backbone_rname = "backbone" if backbone is not None else ""

        # Get neck/FPN
        neck = getattr(module, "neck", getattr(module, "fpn_neck", None))
        neck_rname = "neck" if hasattr(module, "neck") else ("fpn_neck" if hasattr(module, "fpn_neck") else "")

        return Sam2VisionEncoderStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            backbone=backbone,
            backbone_rname=backbone_rname,
            neck=neck,
            neck_rname=neck_rname,
        )


@dataclass(kw_only=True)
class Sam2ModelStruct(BaseModuleStruct):
    """SAM2 Model structure for quantization."""

    module: nn.Module = field(repr=False, kw_only=False)
    parent: None = field(repr=False, default=None)

    # Vision encoder
    vision_encoder: nn.Module = field(repr=False)
    vision_encoder_rname: str

    # Vision encoder struct
    vision_encoder_struct: Sam2VisionEncoderStruct = field(init=False, repr=False)

    # Pre/post module structs (for compatibility with diffusion pattern)
    pre_module_structs: OrderedDict[str, Sam2ModuleStruct] = field(init=False, repr=False)
    post_module_structs: OrderedDict[str, Sam2ModuleStruct] = field(init=False, repr=False)

    @property
    def num_blocks(self) -> int:
        return self.vision_encoder_struct.num_blocks

    @property
    def block_structs(self) -> list[Sam2HieraBlockStruct]:
        return self.vision_encoder_struct.backbone_struct.block_structs

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pre_module_structs = OrderedDict()
        self.post_module_structs = OrderedDict()

        self.vision_encoder_struct = Sam2VisionEncoderStruct.construct(
            self.vision_encoder,
            parent=self,
            fname="vision_encoder",
            rname=self.vision_encoder_rname,
            rkey="vision_encoder",
        )

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        yield from self.vision_encoder_struct.named_key_modules()

    def iter_transformer_block_structs(self) -> tp.Generator[Sam2HieraBlockStruct, None, None]:
        yield from self.vision_encoder_struct.iter_transformer_block_structs()

    def iter_attention_structs(self) -> tp.Generator[Sam2AttentionStruct, None, None]:
        yield from self.vision_encoder_struct.iter_attention_structs()

    def get_prev_module_keys(self) -> tuple[str, ...]:
        return ()

    def get_post_module_keys(self) -> tuple[str, ...]:
        return ()

    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[Sam2HieraBlockStruct], list[bool], list[bool]]:
        layers, layer_structs, recomputes, use_prev_layer_outputs = [], [], [], []
        for block_struct in self.block_structs:
            layers.append(block_struct.module)
            layer_structs.append(block_struct)
            recomputes.append(False)
            use_prev_layer_outputs.append(False)
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    def get_iter_layer_activations_args(
        self, skip_pre_modules: bool, skip_post_modules: bool, **input_kwargs
    ) -> tuple[list[nn.Module], list[Sam2HieraBlockStruct], list[bool], list[bool]]:
        return self._get_iter_block_activations_args(**input_kwargs)

    def get_named_layers(
        self, skip_pre_modules: bool, skip_post_modules: bool, skip_blocks: bool = False
    ) -> OrderedDict[str, Sam2HieraBlockStruct]:
        named_layers = OrderedDict()
        if not skip_blocks:
            for block in self.block_structs:
                named_layers[block.name] = block
        return named_layers

    @staticmethod
    def _default_construct(
        module: nn.Module,
        /,
        parent: None = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2ModelStruct":
        """Construct Sam2ModelStruct from a SAM2 model."""
        # Handle Sam2Model from transformers
        vision_encoder = getattr(module, "vision_encoder", None)
        if vision_encoder is None:
            # Try Sam2VisionModel directly
            if hasattr(module, "backbone"):
                vision_encoder = module
                vision_encoder_rname = ""
            else:
                raise ValueError(f"Cannot find vision encoder in {type(module)}")
        else:
            vision_encoder_rname = "vision_encoder"

        return Sam2ModelStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            vision_encoder=vision_encoder,
            vision_encoder_rname=vision_encoder_rname,
        )

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        /,
        parent: None = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "Sam2ModelStruct":
        """Construct a Sam2ModelStruct from a model."""
        return cls._default_construct(
            module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default key map for quantization."""
        key_map: dict[str, set[str]] = defaultdict(set)
        block_key_map = Sam2HieraBlockStruct._get_default_key_map()
        for rkey, keys in block_key_map.items():
            key_map[rkey].update(keys)
        return {k: v for k, v in key_map.items() if v}

    @staticmethod
    def _simplify_keys(keys: tp.Iterable[str], *, key_map: dict[str, set[str]]) -> list[str]:
        """Simplify keys based on the key map."""
        key_map = dict(sorted(key_map.items(), key=lambda item: len(item[1]), reverse=True))
        ukeys, skeys = set(keys), set()
        for k, v in key_map.items():
            if k in ukeys:
                skeys.add(k)
                ukeys.discard(k)
                ukeys.difference_update(v)
                continue
            if ukeys.issuperset(v):
                skeys.add(k)
                ukeys.difference_update(v)
        assert not ukeys, f"Unrecognized keys: {ukeys}"
        return sorted(skeys)


# Register factories
    # Sam2Model,
    # Sam2VisionModel, 
    # Sam2HieraDetModel,
    # Sam2MultiScaleBlock,
    # Sam2MultiScaleAttention,
    # Sam2FeedForward
Sam2AttentionStruct.register_factory(Sam2MultiScaleAttention, Sam2AttentionStruct._default_construct)
Sam2FeedForwardStruct.register_factory(Sam2FeedForward, Sam2FeedForwardStruct._default_construct)
Sam2HieraBlockStruct.register_factory(Sam2MultiScaleBlock, Sam2HieraBlockStruct._default_construct)
Sam2HieraStruct.register_factory(Sam2HieraDetModel, Sam2HieraStruct._default_construct)
Sam2VisionEncoderStruct.register_factory(Sam2VisionModel, Sam2VisionEncoderStruct._default_construct)
Sam2ModelStruct.register_factory(Sam2Model, Sam2ModelStruct._default_construct)
