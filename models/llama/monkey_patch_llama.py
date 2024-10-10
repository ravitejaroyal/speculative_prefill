"""
    Patch 1: Add monkey patch to llama that allows using the offset position ids
        e.g.
        Original position ids: [0, 1, 2, 3, ..., 15]
        Kept position ids: [0, 1, 4, 8, 13]
        Prefill position ids: [0, 1, 4, 8, 13]
        Decode position ids: [0, 1, 4, 8, 13] + [16, 17, ...]

    Patch 2: Being able to update the full KV cache
        e.g.
        Speculative phase we have KV for [0, 1, 4, 8, 13] + [16, 17, ...]
        After updating we have KV for [0, 1, 2, 3, ..., 15] + [16, 17, ...]
"""

import inspect
import os
from functools import partial
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (GenerationConfig, LlamaForCausalLM, PretrainedConfig,
                          PreTrainedModel)
from transformers.cache_utils import Cache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput, ModelOutput

from models.speculator import (build_speculator, spec_prefill_data_to_inputs,
                               speculate_tokens)


def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        # update the position ids and cache positions correctly here
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            # [B, 1]
            last_pos_ids = model_kwargs["position_ids"][:, -1:]
            model_kwargs["position_ids"] = torch.where(
                last_pos_ids < model_kwargs["original_seq_len"], 
                model_kwargs["original_seq_len"], 
                last_pos_ids + 1
            )
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

            last_pos_ids = model_kwargs["position_ids"][:, -1:]
            model_kwargs["position_ids"] = torch.cat([
                model_kwargs["position_ids"], 
                torch.where(
                    last_pos_ids < model_kwargs["original_seq_len"], 
                    model_kwargs["original_seq_len"], 
                    last_pos_ids + 1
                )
            ], dim=-1)

        return model_kwargs


def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(
                f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                "check the model documentation for supported cache formats."
            )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

            # allow assistant_encoder_outputs to be passed if we're doing assisted generating
            if "assistant_encoder_outputs" in model_kwargs:
                model_args |= {"assistant_encoder_outputs"}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args and key != 'original_seq_len':
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )


def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    speculator = getattr(self, "speculator")
    main_generate = getattr(self, "_generate")

    assert speculator is not None, "Missing speculator"
    assert main_generate is not None, "Missing original generation"

    input_ids = kwargs.pop("input_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    look_ahead_cnt=kwargs.pop("look_ahead_cnt", 8)
    keep=kwargs.pop("keep", 0.3)

    assert input_ids.shape[0] == 1, "Currently only allowing batch size = 1"

    spec_prefill_data = speculate_tokens(
        speculator=speculator, 
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        look_ahead_cnt=look_ahead_cnt, 
        keep=keep
    )

    spec_prefill_inputs = spec_prefill_data_to_inputs(
        spec_prefill_data=spec_prefill_data, 
        input_ids=input_ids, 
        attention_mask=attention_mask
    )

    shrinked_input_len = spec_prefill_inputs["input_ids"].shape[1]

    # get rid of warning
    if "max_new_tokens" in kwargs:
        kwargs.pop("max_length", None)

    if "max_length" in kwargs:
        original_max_length = kwargs["max_length"]
        original_seq_len = input_ids.shape[-1]
        max_gen_len = original_max_length - original_seq_len
        new_max_length = shrinked_input_len + max_gen_len
        kwargs["max_length"] = new_max_length

    outputs = main_generate(
        **spec_prefill_inputs, 
        inputs=inputs,
        generation_config=generation_config,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        synced_gpus=synced_gpus,
        assistant_model=assistant_model,
        streamer=streamer,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        **kwargs,
    )

    # reconstruct the sequences
    # TODO: things other than sequences might not work
    if isinstance(outputs, torch.Tensor):
        generated_tokens = outputs[:, shrinked_input_len:]
        outputs = torch.concatenate(
            [input_ids, generated_tokens], dim=-1
        )
    else:
        generated_tokens = outputs.sequences[:, shrinked_input_len:]
        outputs.sequences = torch.concatenate(
            [input_ids, generated_tokens], dim=-1
        )

    return outputs


def from_pretrained(
    cls,
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    *model_args,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = None,
    **kwargs,
) -> PreTrainedModel:
    original_from_pretrained = kwargs.pop("original_from_pretrained", None)
    assert original_from_pretrained is not None

    model: LlamaForCausalLM = original_from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, 
        *model_args, 
        config=config, 
        cache_dir=cache_dir, 
        ignore_mismatched_sizes=ignore_mismatched_sizes, 
        force_download=force_download, 
        local_files_only=local_files_only, 
        token=token, 
        revision=revision, 
        use_safetensors=use_safetensors, 
        **kwargs
    )

    # restore back the original from pretrained method
    LlamaForCausalLM.from_pretrained = original_from_pretrained
    model.speculator = build_speculator(device=model.device)

    # monkey patch on overriding behaviors
    model._validate_model_kwargs = MethodType(_validate_model_kwargs, model)
    model._update_model_kwargs_for_generation = MethodType(_update_model_kwargs_for_generation, model)
    
    # monkey patch on modifying behaviors
    model._generate = model.generate
    model.generate = MethodType(torch.no_grad(generate), model)

    return model


def monkey_patch_llama():
    original_from_pretrained = LlamaForCausalLM.from_pretrained
    LlamaForCausalLM.from_pretrained = classmethod(partial(
        from_pretrained, 
        original_from_pretrained=original_from_pretrained
    ))
