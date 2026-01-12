# -*- coding: utf-8 -*-
"""Utility functions for SAM2 (Segment Anything Model 2) from HuggingFace."""

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
    "Sam2ConfigStruct",
    "Sam2ModelStruct",
    "Sam2VisionEncoderStruct",
    "Sam2HieraBlockStruct",
    "Sam2AttentionStruct",
    "Sam2FeedForwardStruct",
]


@dataclass(kw_only=True)
class Sam2ConfigStruct(AttentionConfigStruct, FeedForwardConfigStruct):
    """SAM2 Configuration for HuggingFace models.

    Args:
        hidden_size (`int`):
            The size of the hidden representations.
        intermediate_size (`int`):
            The size of the intermediate (MLP) layer.
        num_hidden_layers (`int`):
            The number of Hiera blocks in the vision encoder.
        num_attention_heads (`int`):
            The number of attention heads.
        image_size (`int`):
            The size of input images.
        patch_size (`int`):
            The size of each patch.
        num_channels (`int`):
            The number of input channels.
        embed_dim (`int`):
            The embedding dimension (same as hidden_size for Hiera).
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int = 1024
    patch_size: int = 16
    num_channels: int = 3
    embed_dim: int = 768


@dataclass(kw_only=True)
class Sam2ModuleStruct(BaseModuleStruct):
    """Base SAM2 module structure."""

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
class Sam2AttentionStruct(AttentionStruct, Sam2ModuleStruct):
    """SAM2 Attention Block (supports Hiera multi-head attention)."""

    # relative keys for Q, K, V projections
    q_rkey: tp.ClassVar[str] = "attn_q"
    k_rkey: tp.ClassVar[str] = "attn_k"
    v_rkey: tp.ClassVar[str] = "attn_v"
    o_rkey: tp.ClassVar[str] = "attn_o"

    parent: tp.Optional["Sam2HieraBlockStruct"] = field(repr=False)
    qkv: nn.Module | None = field(default=None, repr=False)  # Combined QKV projection
    q_proj: nn.Module | None = field(default=None, repr=False)
    k_proj: nn.Module | None = field(default=None, repr=False)
    v_proj: nn.Module | None = field(default=None, repr=False)
    o_proj: nn.Module | None = field(default=None, repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: Sam2ConfigStruct,
        parent: tp.Optional["Sam2HieraBlockStruct"] = None,
    ) -> "Sam2AttentionStruct":
        """Construct Sam2AttentionStruct from attention module."""
        # SAM2 Hiera typically has:
        # - qkv: combined projection for q, k, v
        # - proj: output projection
        qkv = getattr(module, "qkv", None)
        q_proj = getattr(module, "q_proj", getattr(module, "q", None))
        k_proj = getattr(module, "k_proj", getattr(module, "k", None))
        v_proj = getattr(module, "v_proj", getattr(module, "v", None))
        o_proj = getattr(module, "proj", getattr(module, "o_proj", getattr(module, "out_proj", None)))

        return cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            qkv=qkv,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )


@dataclass(kw_only=True)
class Sam2FeedForwardStruct(FeedForwardStruct, Sam2ModuleStruct):
    """SAM2 Feed-Forward Block (MLP in Hiera)."""

    # relative keys for feed-forward layers
    fc1_rkey: tp.ClassVar[str] = "ffn_fc1"
    fc2_rkey: tp.ClassVar[str] = "ffn_fc2"

    parent: tp.Optional["Sam2HieraBlockStruct"] = field(repr=False)
    fc1: nn.Module | None = field(default=None, repr=False)
    fc2: nn.Module | None = field(default=None, repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: Sam2ConfigStruct,
        parent: tp.Optional["Sam2HieraBlockStruct"] = None,
    ) -> "Sam2FeedForwardStruct":
        """Construct Sam2FeedForwardStruct from MLP module."""
        # Different naming conventions in SAM2/Hiera
        fc1 = getattr(module, "fc1", getattr(module, "lin1", getattr(module, "c_fc", None)))
        fc2 = getattr(module, "fc2", getattr(module, "lin2", getattr(module, "c_proj", None)))

        return cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            fc1=fc1,
            fc2=fc2,
        )


@dataclass(kw_only=True)
class Sam2HieraBlockStruct(TransformerBlockStruct, Sam2ModuleStruct):
    """SAM2 Hiera Transformer Block."""

    parent: tp.Optional["Sam2VisionEncoderStruct"] = field(repr=False)
    attn: Sam2AttentionStruct | None = field(default=None, repr=False)
    mlp: Sam2FeedForwardStruct | None = field(default=None, repr=False)
    norm1: nn.Module | None = field(default=None, repr=False)
    norm2: nn.Module | None = field(default=None, repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: Sam2ConfigStruct,
        parent: tp.Optional["Sam2VisionEncoderStruct"] = None,
    ) -> "Sam2HieraBlockStruct":
        """Construct Sam2HieraBlockStruct from Hiera block."""
        # Get attention module
        attn_module = getattr(module, "attn", getattr(module, "self_attn", getattr(module, "attention", None)))
        attn = None
        if attn_module is not None:
            attn = Sam2AttentionStruct.construct(
                attn_module,
                join_name(name, "attn"),
                config,
                parent=None,
            )

        # Get MLP module
        mlp_module = getattr(module, "mlp", getattr(module, "ffn", getattr(module, "feed_forward", None)))
        mlp = None
        if mlp_module is not None:
            mlp = Sam2FeedForwardStruct.construct(
                mlp_module,
                join_name(name, "mlp"),
                config,
                parent=None,
            )

        # Get normalization layers
        norm1 = getattr(module, "norm1", getattr(module, "ln_1", getattr(module, "layer_norm1", None)))
        norm2 = getattr(module, "norm2", getattr(module, "ln_2", getattr(module, "layer_norm2", None)))

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
        if attn is not None:
            attn.parent = block
        if mlp is not None:
            mlp.parent = block

        return block

    def iter_attention_structs(self) -> tp.Generator[Sam2AttentionStruct, None, None]:
        """Iterate over attention structures in this block."""
        if self.attn is not None:
            yield self.attn

    def iter_ffn_structs(self) -> tp.Generator[Sam2FeedForwardStruct, None, None]:
        """Iterate over feed-forward structures in this block."""
        if self.mlp is not None:
            yield self.mlp


@dataclass(kw_only=True)
class Sam2VisionEncoderStruct(BaseTransformerStruct, Sam2ModuleStruct):
    """SAM2 Vision Encoder (Hiera backbone)."""

    parent: tp.Optional["Sam2ModelStruct"] = field(repr=False)
    blocks: list[Sam2HieraBlockStruct] = field(default_factory=list, repr=False)
    patch_embed: nn.Module | None = field(default=None, repr=False)
    pos_embed: nn.Module | None = field(default=None, repr=False)
    norm: nn.Module | None = field(default=None, repr=False)

    @classmethod
    def construct(
        cls,
        module: nn.Module,
        name: str,
        config: Sam2ConfigStruct,
        parent: tp.Optional["Sam2ModelStruct"] = None,
    ) -> "Sam2VisionEncoderStruct":
        """Construct Sam2VisionEncoderStruct from vision encoder module."""
        # Get patch embedding (Hiera uses patch_embed)
        patch_embed = getattr(module, "patch_embed", getattr(module, "embeddings", None))

        # Get positional embedding
        pos_embed = getattr(module, "pos_embed", None)

        # Get Hiera blocks
        blocks = []
        # Try different possible locations for blocks
        blocks_container = getattr(module, "blocks", getattr(module, "layers", getattr(module, "encoder", None)))

        if blocks_container is not None:
            # If blocks_container is a module with layers/blocks, get them
            if hasattr(blocks_container, "layers"):
                blocks_list = blocks_container.layers
            elif hasattr(blocks_container, "blocks"):
                blocks_list = blocks_container.blocks
            elif isinstance(blocks_container, (list, nn.ModuleList)):
                blocks_list = blocks_container
            else:
                blocks_list = []

            for idx, block_module in enumerate(blocks_list):
                block = Sam2HieraBlockStruct.construct(
                    block_module,
                    join_name(name, f"blocks.{idx}"),
                    config,
                    parent=None,
                )
                blocks.append(block)

        # Get final normalization
        norm = getattr(module, "norm", getattr(module, "ln_post", getattr(module, "final_layer_norm", None)))

        encoder = cls(
            module=module,
            name=name,
            parent=parent,
            config=config,
            blocks=blocks,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            norm=norm,
        )

        # Set parent references
        for block in blocks:
            block.parent = encoder

        return encoder

    @property
    def num_blocks(self) -> int:
        """Number of Hiera blocks."""
        return len(self.blocks)

    def iter_block_structs(self) -> tp.Generator[Sam2HieraBlockStruct, None, None]:
        """Iterate over Hiera block structures."""
        yield from self.blocks

    def iter_attention_structs(self) -> tp.Generator[Sam2AttentionStruct, None, None]:
        """Iterate over all attention structures in the encoder."""
        for block in self.blocks:
            yield from block.iter_attention_structs()

    def iter_ffn_structs(self) -> tp.Generator[Sam2FeedForwardStruct, None, None]:
        """Iterate over all feed-forward structures in the encoder."""
        for block in self.blocks:
            yield from block.iter_ffn_structs()


@dataclass(kw_only=True)
class Sam2ModelStruct(Sam2ModuleStruct):
    """SAM2 Model Structure for HuggingFace models."""

    config: Sam2ConfigStruct
    vision_encoder: Sam2VisionEncoderStruct = field(repr=False)
    mask_decoder: nn.Module | None = field(default=None, repr=False)
    prompt_encoder: nn.Module | None = field(default=None, repr=False)
    pre_module_structs: OrderedDict[str, Sam2ModuleStruct] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    post_module_structs: OrderedDict[str, Sam2ModuleStruct] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    @classmethod
    def construct(
        cls,
        model: nn.Module,
        image_size: int = 1024,
        patch_size: int = 16,
    ) -> "Sam2ModelStruct":
        """Construct Sam2ModelStruct from SAM2 HuggingFace model.

        Args:
            model (`nn.Module`):
                The SAM2 model instance from HuggingFace (e.g., Sam2Model).
            image_size (`int`, *optional*, defaults to 1024):
                Input image size.
            patch_size (`int`, *optional*, defaults to 16):
                Patch size for vision transformer.

        Returns:
            `Sam2ModelStruct`: The constructed SAM2 model structure.
        """
        # Extract configuration from HuggingFace model
        if hasattr(model, "config"):
            hf_config = model.config
            # SAM2 has vision_config for the Hiera backbone
            vision_config = hf_config.vision_config if hasattr(hf_config, "vision_config") else hf_config

            hidden_size = getattr(vision_config, "hidden_size", getattr(vision_config, "embed_dim", 768))
            intermediate_size = getattr(
                vision_config, "intermediate_size", getattr(vision_config, "mlp_dim", hidden_size * 4)
            )
            num_hidden_layers = getattr(vision_config, "num_hidden_layers", getattr(vision_config, "depth", 12))
            num_attention_heads = getattr(
                vision_config, "num_attention_heads", getattr(vision_config, "num_heads", 12)
            )
            image_size = getattr(vision_config, "image_size", image_size)
            patch_size = getattr(vision_config, "patch_size", patch_size)
            embed_dim = getattr(vision_config, "embed_dim", hidden_size)
        else:
            # Default SAM2-Hiera-Tiny configuration
            hidden_size = 768
            intermediate_size = 3072
            num_hidden_layers = 12
            num_attention_heads = 12
            embed_dim = 768

        config = Sam2ConfigStruct(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            embed_dim=embed_dim,
        )

        # Get vision encoder from SAM2 model
        vision_encoder_module = None
        if hasattr(model, "vision_encoder"):
            vision_encoder_module = model.vision_encoder
        elif hasattr(model, "image_encoder"):
            vision_encoder_module = model.image_encoder
        elif hasattr(model, "sam"):
            if hasattr(model.sam, "vision_encoder"):
                vision_encoder_module = model.sam.vision_encoder
            elif hasattr(model.sam, "image_encoder"):
                vision_encoder_module = model.sam.image_encoder
        else:
            # Fallback: use the whole model
            vision_encoder_module = model

        vision_encoder = Sam2VisionEncoderStruct.construct(
            vision_encoder_module,
            "vision_encoder",
            config,
            parent=None,
        )

        # Get mask decoder (if exists)
        mask_decoder = getattr(model, "mask_decoder", None)

        # Get prompt encoder (if exists)
        prompt_encoder = getattr(model, "prompt_encoder", None)

        sam2_struct = cls(
            module=model,
            name="",
            parent=None,
            config=config,
            vision_encoder=vision_encoder,
            mask_decoder=mask_decoder,
            prompt_encoder=prompt_encoder,
        )

        # Set parent reference
        vision_encoder.parent = sam2_struct

        return sam2_struct

    @property
    def num_blocks(self) -> int:
        """Total number of Hiera blocks."""
        return self.vision_encoder.num_blocks

    def iter_block_structs(self) -> tp.Generator[Sam2HieraBlockStruct, None, None]:
        """Iterate over all Hiera block structures."""
        yield from self.vision_encoder.iter_block_structs()

    def iter_attention_structs(self) -> tp.Generator[Sam2AttentionStruct, None, None]:
        """Iterate over all attention structures."""
        yield from self.vision_encoder.iter_attention_structs()

    def iter_ffn_structs(self) -> tp.Generator[Sam2FeedForwardStruct, None, None]:
        """Iterate over all feed-forward structures."""
        yield from self.vision_encoder.iter_ffn_structs()

    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        """Iterate over all named quantizable modules."""
        # Vision encoder modules
        yield from self.vision_encoder.named_key_modules()

        # Pre and post modules
        for struct in self.pre_module_structs.values():
            yield from struct.named_key_modules()
        for struct in self.post_module_structs.values():
            yield from struct.named_key_modules()
