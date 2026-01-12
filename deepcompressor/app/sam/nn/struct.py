# -*- coding: utf-8 -*-
"""Utility functions for SAM (Segment Anything Model)."""

import typing as tp
from collections import OrderedDict
from dataclasses import dataclass, field

import torch.nn as nn

from deepcompressor.nn.struct.attn import (
    AttentionConfigStruct,
    AttentionStruct,
    BaseTransformerStruct,
    FeedForwardConfigStruct,
    FeedForwardStruct,
    TransformerBlockStruct,
)
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

__all__ = [
    "SamConfigStruct",
    "SamModelStruct",
    "SamBackboneStruct",
    "SamNunchakuBlockStruct",
    "SamAttentionStruct",
    "SamFeedForwardStruct",
    "SamDecoderStruct",
]


@dataclass(kw_only=True)
class SamConfigStruct(AttentionConfigStruct, FeedForwardConfigStruct):
    """SAM Configuration.

    Args:
        hidden_size (`int`):
            The size of the hidden representations.
        intermediate_size (`int`):
            The size of the intermediate (MLP) layer.
        num_hidden_layers (`int`):
            The number of transformer blocks in the backbone.
        num_attention_heads (`int`):
            The number of attention heads.
        patch_size (`int`):
            The size of each patch.
        image_size (`int`):
            The size of input images.
        num_channels (`int`):
            The number of input channels.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int = 16
    image_size: int = 1024
    num_channels: int = 3


@dataclass(kw_only=True)
class SamModuleStruct(BaseModuleStruct):
    """Base SAM module structure."""

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        if isinstance(self.module, (nn.Linear, nn.Conv2d)):
            yield self.key, self.name, self.module, self.parent, self.fname
        else:
            for name, module in self.module.named_modules():
                if name and isinstance(module, (nn.Linear, nn.Conv2d)):
                    module_name = join_name(self.name, name)
                    field_name = join_name(self.fname, name)
                    yield self.key, module_name, module, self.parent, field_name


@dataclass(kw_only=True)
class SamAttentionStruct(AttentionStruct, SamModuleStruct):
    """SAM Attention Block."""

    # relative keys for Q, K, V projections
    q_rkey: tp.ClassVar[str] = "attn_q"
    k_rkey: tp.ClassVar[str] = "attn_k"
    v_rkey: tp.ClassVar[str] = "attn_v"
    o_rkey: tp.ClassVar[str] = "attn_o"

    parent: tp.Optional["SamNunchakuBlockStruct"] = field(repr=False)
    q_proj: nn.Linear = field(repr=False)
    k_proj: nn.Linear = field(repr=False)
    v_proj: nn.Linear = field(repr=False)
    o_proj: nn.Linear = field(repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: SamConfigStruct,
        parent: tp.Optional["SamNunchakuBlockStruct"] = None,
    ) -> "SamAttentionStruct":
        """Construct SamAttentionStruct from attention module."""
        # Handle different SAM attention implementations
        # Standard SAM2/Nunchaku has qkv as single projection or separate projections
        if hasattr(module, "qkv"):
            # Single projection for q, k, v
            q_proj = k_proj = v_proj = module.qkv
        else:
            # Separate projections
            q_proj = getattr(module, "q_proj", getattr(module, "q", None))
            k_proj = getattr(module, "k_proj", getattr(module, "k", None))
            v_proj = getattr(module, "v_proj", getattr(module, "v", None))

        o_proj = getattr(module, "proj", getattr(module, "o_proj", None))

        return cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )


@dataclass(kw_only=True)
class SamFeedForwardStruct(FeedForwardStruct, SamModuleStruct):
    """SAM Feed-Forward Block (MLP)."""

    # relative keys for feed-forward layers
    gate_rkey: tp.ClassVar[str] = "ffn_gate"
    up_rkey: tp.ClassVar[str] = "ffn_up"
    down_rkey: tp.ClassVar[str] = "ffn_down"

    parent: tp.Optional["SamNunchakuBlockStruct"] = field(repr=False)
    fc1: nn.Linear = field(repr=False)
    fc2: nn.Linear = field(repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: SamConfigStruct,
        parent: tp.Optional["SamNunchakuBlockStruct"] = None,
    ) -> "SamFeedForwardStruct":
        """Construct SamFeedForwardStruct from MLP module."""
        # Different naming conventions
        fc1 = getattr(module, "fc1", getattr(module, "lin1", getattr(module, "w1", None)))
        fc2 = getattr(module, "fc2", getattr(module, "lin2", getattr(module, "w2", None)))

        return cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            fc1=fc1,
            fc2=fc2,
        )


@dataclass(kw_only=True)
class SamNunchakuBlockStruct(TransformerBlockStruct, SamModuleStruct):
    """SAM Nunchaku Transformer Block."""

    parent: tp.Optional["SamBackboneStruct"] = field(repr=False)
    attn: SamAttentionStruct = field(repr=False)
    mlp: SamFeedForwardStruct = field(repr=False)
    norm1: nn.Module = field(repr=False)
    norm2: nn.Module = field(repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: SamConfigStruct,
        parent: tp.Optional["SamBackboneStruct"] = None,
    ) -> "SamNunchakuBlockStruct":
        """Construct SamNunchakuBlockStruct from transformer block."""
        # Get attention module
        attn_module = getattr(module, "attn", getattr(module, "self_attn", None))
        attn = SamAttentionStruct.construct(
            attn_module,
            join_name(name, "attn"),
            config,
            parent=None,  # Will set later
        )

        # Get MLP module
        mlp_module = getattr(module, "mlp", getattr(module, "ffn", None))
        mlp = SamFeedForwardStruct.construct(
            mlp_module,
            join_name(name, "mlp"),
            config,
            parent=None,  # Will set later
        )

        # Get normalization layers
        norm1 = getattr(module, "norm1", getattr(module, "ln_1", None))
        norm2 = getattr(module, "norm2", getattr(module, "ln_2", None))

        block = cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            attn=attn,
            mlp=mlp,
            norm1=norm1,
            norm2=norm2,
        )

        # Set parent references
        attn.parent = block
        mlp.parent = block

        return block

    def iter_attention_structs(self) -> tp.Generator[SamAttentionStruct, None, None]:
        """Iterate over attention structures in this block."""
        yield self.attn

    def iter_ffn_structs(self) -> tp.Generator[SamFeedForwardStruct, None, None]:
        """Iterate over feed-forward structures in this block."""
        yield self.mlp


@dataclass(kw_only=True)
class SamBackboneStruct(BaseTransformerStruct, SamModuleStruct):
    """SAM Nunchaku Backbone (Vision Transformer)."""

    parent: tp.Optional["SamModelStruct"] = field(repr=False)
    blocks: list[SamNunchakuBlockStruct] = field(default_factory=list, repr=False)
    patch_embed: nn.Module = field(repr=False)
    norm: nn.Module | None = field(default=None, repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: SamConfigStruct,
        parent: tp.Optional["SamModelStruct"] = None,
    ) -> "SamBackboneStruct":
        """Construct SamBackboneStruct from backbone module."""
        # Get patch embedding
        patch_embed = getattr(module, "patch_embed", getattr(module, "pos_embed", None))

        # Get transformer blocks
        blocks = []
        blocks_container = getattr(module, "blocks", getattr(module, "layers", []))

        for idx, block_module in enumerate(blocks_container):
            block = SamNunchakuBlockStruct.construct(
                block_module,
                join_name(name, f"blocks.{idx}"),
                config,
                parent=None,  # Will set later
            )
            blocks.append(block)

        # Get final normalization
        norm = getattr(module, "norm", getattr(module, "ln_f", None))

        backbone = cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            blocks=blocks,
            patch_embed=patch_embed,
            norm=norm,
        )

        # Set parent references
        for block in blocks:
            block.parent = backbone

        return backbone

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks."""
        return len(self.blocks)

    def iter_block_structs(self) -> tp.Generator[SamNunchakuBlockStruct, None, None]:
        """Iterate over transformer block structures."""
        yield from self.blocks

    def iter_attention_structs(self) -> tp.Generator[SamAttentionStruct, None, None]:
        """Iterate over all attention structures in the backbone."""
        for block in self.blocks:
            yield from block.iter_attention_structs()

    def iter_ffn_structs(self) -> tp.Generator[SamFeedForwardStruct, None, None]:
        """Iterate over all feed-forward structures in the backbone."""
        for block in self.blocks:
            yield from block.iter_ffn_structs()


@dataclass(kw_only=True)
class SamDecoderStruct(SamModuleStruct):
    """SAM Mask Decoder."""

    parent: tp.Optional["SamModelStruct"] = field(repr=False)
    transformer_blocks: list[nn.Module] = field(default_factory=list, repr=False)
    iou_prediction_head: nn.Module | None = field(default=None, repr=False)
    mask_tokens: nn.Module | None = field(default=None, repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: SamConfigStruct,
        parent: tp.Optional["SamModelStruct"] = None,
    ) -> "SamDecoderStruct":
        """Construct SamDecoderStruct from decoder module."""
        # Get transformer blocks in decoder
        transformer_blocks = []
        if hasattr(module, "transformer"):
            if hasattr(module.transformer, "layers"):
                transformer_blocks = list(module.transformer.layers)
            elif hasattr(module.transformer, "blocks"):
                transformer_blocks = list(module.transformer.blocks)

        # Get prediction heads
        iou_head = getattr(module, "iou_prediction_head", None)
        mask_tokens = getattr(module, "mask_tokens", None)

        return cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            transformer_blocks=transformer_blocks,
            iou_prediction_head=iou_head,
            mask_tokens=mask_tokens,
        )


@dataclass(kw_only=True)
class SamModelStruct(SamModuleStruct):
    """SAM Model Structure."""

    config: SamConfigStruct
    backbone: SamBackboneStruct = field(repr=False)
    decoder: SamDecoderStruct | None = field(default=None, repr=False)
    prompt_encoder: nn.Module | None = field(default=None, repr=False)
    pre_module_structs: OrderedDict[str, SamModuleStruct] = field(default_factory=OrderedDict, init=False, repr=False)
    post_module_structs: OrderedDict[str, SamModuleStruct] = field(default_factory=OrderedDict, init=False, repr=False)

    @classmethod
    def construct(
        cls,
        model: nn.Module,
        image_size: int = 1024,
        patch_size: int = 16,
    ) -> "SamModelStruct":
        """Construct SamModelStruct from SAM model.

        Args:
            model (`nn.Module`):
                The SAM model instance (supports SAM2 from HuggingFace).
            image_size (`int`, *optional*, defaults to 1024):
                Input image size.
            patch_size (`int`, *optional*, defaults to 16):
                Patch size for vision transformer.

        Returns:
            `SamModelStruct`: The constructed SAM model structure.
        """
        # Extract configuration from model
        if hasattr(model, "config"):
            model_config = model.config
            # Handle nested vision_config (common in HuggingFace SAM2)
            vision_config = (
                model_config.vision_config if hasattr(model_config, "vision_config") else model_config
            )
            hidden_size = getattr(vision_config, "hidden_size", 768)
            intermediate_size = getattr(vision_config, "intermediate_size", 3072)
            num_hidden_layers = getattr(vision_config, "num_hidden_layers", 12)
            num_attention_heads = getattr(vision_config, "num_attention_heads", 12)
            # Try to get image_size and patch_size from config
            image_size = getattr(vision_config, "image_size", image_size)
            patch_size = getattr(vision_config, "patch_size", patch_size)
        else:
            # Default SAM-B configuration
            hidden_size = 768
            intermediate_size = 3072
            num_hidden_layers = 12
            num_attention_heads = 12

        config = SamConfigStruct(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            patch_size=patch_size,
            image_size=image_size,
            num_channels=3,
        )

        # Get image encoder (backbone) - handle different SAM2 architectures
        image_encoder = None
        # Try different possible locations for vision/image encoder
        if hasattr(model, "vision_encoder"):
            image_encoder = model.vision_encoder
        elif hasattr(model, "image_encoder"):
            image_encoder = model.image_encoder
        elif hasattr(model, "sam"):
            if hasattr(model.sam, "vision_encoder"):
                image_encoder = model.sam.vision_encoder
            elif hasattr(model.sam, "image_encoder"):
                image_encoder = model.sam.image_encoder

        # Fallback to using the entire model if no encoder found
        if image_encoder is None:
            image_encoder = model
        backbone = SamBackboneStruct.construct(
            image_encoder,
            "image_encoder",
            config,
            parent=None,  # Will set later
        )

        # Get mask decoder (if exists)
        decoder = None
        if hasattr(model, "mask_decoder"):
            decoder = SamDecoderStruct.construct(
                model.mask_decoder,
                "mask_decoder",
                config,
                parent=None,  # Will set later
            )

        # Get prompt encoder (if exists)
        prompt_encoder = getattr(model, "prompt_encoder", None)

        sam_struct = cls(
            module=model,
            name="",
            parent=None,
            config=config,
            backbone=backbone,
            decoder=decoder,
            prompt_encoder=prompt_encoder,
        )

        # Set parent references
        backbone.parent = sam_struct
        if decoder is not None:
            decoder.parent = sam_struct

        return sam_struct

    @property
    def num_blocks(self) -> int:
        """Total number of transformer blocks."""
        return self.backbone.num_blocks

    def iter_block_structs(self) -> tp.Generator[SamNunchakuBlockStruct, None, None]:
        """Iterate over all transformer block structures."""
        yield from self.backbone.iter_block_structs()

    def iter_attention_structs(self) -> tp.Generator[SamAttentionStruct, None, None]:
        """Iterate over all attention structures."""
        yield from self.backbone.iter_attention_structs()

    def iter_ffn_structs(self) -> tp.Generator[SamFeedForwardStruct, None, None]:
        """Iterate over all feed-forward structures."""
        yield from self.backbone.iter_ffn_structs()

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        """Iterate over all named quantizable modules."""
        # Backbone modules
        yield from self.backbone.named_key_modules()

        # Decoder modules (if exists)
        if self.decoder is not None:
            yield from self.decoder.named_key_modules()

        # Pre and post modules
        for struct in self.pre_module_structs.values():
            yield from struct.named_key_modules()
        for struct in self.post_module_structs.values():
            yield from struct.named_key_modules()
