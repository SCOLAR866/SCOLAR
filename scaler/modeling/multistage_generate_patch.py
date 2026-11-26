"""
Multi-stage generate() patch for Qwen2_5_VLForConditionalGeneration.

Design
------
After this patch, a single ``model.generate(...)`` call automatically runs
the two-phase "auxiliary visual token" generation paradigm:

    Phase 1 (LM text, trigger hunt):
        Normal HF generation, but ``<auxiliary>`` is added to the eos set.
        Generation stops when the model either emits ``<auxiliary>`` or a
        real eos. This returns ``phase1_ids = prompt + new_text[+<auxiliary>]``.

    Phase 2 (detransformer + LM text, only if <auxiliary> triggered):
        Feed ``phase1_ids`` (which already ends with ``<auxiliary>``) to
        ``pass_through_visual_detransformer`` -> get ``inputs_embeds`` with
        the aux-token block spliced in -> resume HF generation on those
        ``inputs_embeds`` to produce the answer text.

Batching
--------
B == 1 -> single-sample path (clean, zero-copy).
B >  1 -> **real batched** path:
    * phase-1  :  one batched HF generate on left-padded prompts.
    * detrans  :  per-sample (the visual_detransformer is tiny, so looping
                  is fine and avoids padding-direction pitfalls).
    * phase-2  :  left-pad the per-sample ``inputs_embeds`` to the same
                  length, concat, then one batched HF generate over only
                  the samples that actually triggered ``<auxiliary>``.

Auto-routing rules
------------------
    * ``inputs_embeds is not None``            -> vanilla (RL rollout path).
    * ``pixel_values`` / ``image_grid_thw``    -> vanilla (pure-text input).
        is None
    * ``multistage=False``                     -> vanilla (explicit override).
    * ``multistage=True`` or auto + has image  -> two-phase.

The original HF ``generate`` remains reachable as
``model.generate_vanilla(...)``.
"""

from __future__ import annotations

from typing import Any

import torch

from transformers import Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------

def _get_auxiliary_start_id(self) -> int | None:
    explicit = getattr(self.config, "auxiliary_start_token_id", None)
    if explicit is not None:
        return int(explicit)
    dt_cfg = getattr(self.config, "detransformer_config", None)
    if dt_cfg is None:
        return None
    ids = getattr(dt_cfg, "gen_auxiliary_start_token_ids", None)
    if not ids:
        return None
    return int(ids[0])


def _as_list_eos(eos) -> list[int]:
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(x) for x in eos]


def _split_per_sample_visual(pixel_values, image_grid_thw):
    """Split ``pixel_values`` (flattened [sum_patches, C]) per-sample,
    assuming **one image per sample**."""
    assert image_grid_thw is not None
    B = image_grid_thw.shape[0]
    lengths = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
    offset = 0
    out_pv, out_thw = [], []
    for i in range(B):
        L = int(lengths[i])
        out_pv.append(pixel_values[offset: offset + L])
        out_thw.append(image_grid_thw[i: i + 1])
        offset += L
    return out_pv, out_thw


def _ensure_left_pad(ids: torch.Tensor,
                     mask: torch.Tensor,
                     pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize any padding layout to **left-padding** (HF decoder-only
    generate requires left padding).

    Uses the boolean ``mask`` to select the real tokens of each row, so the
    input can be right-padded, left-padded, or unpadded -- all flow through.
    Returns new ``[B, S]`` ids + mask.
    """
    B, S = ids.shape
    out_ids = torch.full_like(ids, pad_id)
    out_mask = torch.zeros_like(mask)
    bool_mask = mask.bool()
    for i in range(B):
        real_tokens = ids[i][bool_mask[i]]
        real_len = real_tokens.shape[0]
        if real_len > 0:
            out_ids[i, S - real_len:] = real_tokens
            out_mask[i, S - real_len:] = 1
    return out_ids, out_mask


def _parse_phase1_per_sample(phase1_out: torch.Tensor,
                             prompt_len_in: int,
                             aux_id: int,
                             eos_ids: list[int]) -> tuple[list[bool], list[int]]:
    """For each sample, locate phase-1's stop column and whether ``<auxiliary>``
    triggered.

    ``phase1_out[:, prompt_len_in:]`` is the newly generated region.
    Returns ``(triggered_list, stop_pos_list)`` where ``stop_pos_list[i]`` is
    the **absolute** (inclusive) column index of the last kept token of
    sample ``i`` in ``phase1_out``.
    """
    B = phase1_out.shape[0]
    new_len = phase1_out.shape[1] - prompt_len_in
    triggered_list: list[bool] = []
    stop_pos: list[int] = []
    for i in range(B):
        row = phase1_out[i, prompt_len_in:].tolist()
        hit = -1
        hit_trig = False
        for k, tok in enumerate(row):
            if tok == aux_id:
                hit, hit_trig = k, True
                break
            if tok in eos_ids:
                hit, hit_trig = k, False
                break
        if hit < 0:
            # No stop token encountered within max_new_phase1.
            hit = new_len - 1
            hit_trig = False
        triggered_list.append(hit_trig)
        stop_pos.append(prompt_len_in + hit)
    return triggered_list, stop_pos


def _right_pad_traj(trajs: list[torch.Tensor],
                    pad_id: int) -> torch.Tensor:
    """Right-pad a list of 1-D LongTensor to a single [B, L_max] tensor."""
    max_len = max(t.shape[0] for t in trajs)
    device = trajs[0].device
    dtype  = trajs[0].dtype
    out = torch.full((len(trajs), max_len), pad_id, dtype=dtype, device=device)
    for i, t in enumerate(trajs):
        out[i, :t.shape[0]] = t
    return out


# ---------------------------------------------------------------------------
# single-sample path  (B == 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _single_sample_two_phase(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.LongTensor,
    *,
    auxiliary_start_id: int,
    generate_type: str,
    max_new_tokens_phase1: int,
    max_new_tokens_phase2: int,
    eos_token_id: list[int],
    pad_token_id: int,
    **gen_kwargs,
) -> torch.LongTensor:
    phase1_eos = list(set([auxiliary_start_id] + eos_token_id))
    phase1_out = self.generate_vanilla(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens_phase1,
        eos_token_id=phase1_eos,
        pad_token_id=pad_token_id,
        **gen_kwargs,
    )
    last_tok = int(phase1_out[0, -1].item())
    if last_tok != auxiliary_start_id:
        return phase1_out

    phase1_mask = torch.cat([attention_mask, torch.ones_like(phase1_out[:, input_ids.shape[1]:])], dim=1)
    phase2_inputs = self.pass_through_visual_detransformer(
        generate_type=generate_type,
        input_ids=phase1_out,
        attention_mask=phase1_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    phase2_out = self.generate_vanilla(
        input_ids=phase2_inputs["input_ids"],
        inputs_embeds=phase2_inputs["inputs_embeds"],
        attention_mask=phase2_inputs["attention_mask"],
        image_grid_thw=phase2_inputs["image_grid_thw"],
        max_new_tokens=max_new_tokens_phase2,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        **gen_kwargs,
    )
    phase2_new = phase2_out[:, phase2_inputs["input_ids"].shape[1]:]
    return torch.cat([phase1_out, phase2_new], dim=1)


# ---------------------------------------------------------------------------
# batched path  (B > 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batched_two_phase(
    self: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.LongTensor,        # [B, S]  right-padded
    attention_mask: torch.Tensor,       # [B, S]  right-padded
    pixel_values: torch.Tensor,
    image_grid_thw: torch.LongTensor,
    *,
    auxiliary_start_id: int,
    generate_type: str,
    max_new_tokens_phase1: int,
    max_new_tokens_phase2: int,
    eos_token_id: list[int],
    pad_token_id: int,
    **gen_kwargs,
) -> torch.LongTensor:
    device = input_ids.device
    B, S = input_ids.shape

    # ---------------- Phase 1: batched generate on left-padded prompts -----
    ids_lp, mask_lp = _ensure_left_pad(input_ids, attention_mask, pad_token_id)

    phase1_eos = list(set([auxiliary_start_id] + eos_token_id))
    phase1_out = self.generate_vanilla(
        input_ids=ids_lp,
        attention_mask=mask_lp,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens_phase1,
        eos_token_id=phase1_eos,
        pad_token_id=pad_token_id,
        **gen_kwargs,
    )
    # phase1_out: [B, S + new_len]
    #   cols [0, S)  = left-padded prompt
    #   cols [S, S+new_len) = newly generated (pad-tail per sample)
    triggered_list, stop_pos_list = _parse_phase1_per_sample(
        phase1_out, prompt_len_in=S, aux_id=auxiliary_start_id, eos_ids=eos_token_id
    )

    # Strip left-padding & right-tail per sample -> list of 1-D tensors.
    trajectories: list[torch.Tensor] = []
    for i in range(B):
        real_prompt_len = int(attention_mask[i].sum().item())
        left_pad_len = S - real_prompt_len
        keep_end = stop_pos_list[i] + 1  # inclusive
        # Drop any pad_id at the very end (in case parse fell through).
        traj = phase1_out[i, left_pad_len:keep_end].clone()
        trajectories.append(traj)

    triggered_idx = [i for i, t in enumerate(triggered_list) if t]

    # ---------------- Phase 2: only on triggered samples -------------------
    if len(triggered_idx) > 0:
        per_sample_pv, per_sample_thw = _split_per_sample_visual(
            pixel_values, image_grid_thw
        )

        # Per-sample detransformer (small model; loop is fine).
        phase2_inputs_list = []
        for i in triggered_idx:
            p1_ids  = trajectories[i].unsqueeze(0).to(device)
            p1_mask = torch.ones_like(p1_ids)
            dt_out = self.pass_through_visual_detransformer(
                generate_type=generate_type,
                input_ids=p1_ids,
                attention_mask=p1_mask,
                pixel_values=per_sample_pv[i],
                image_grid_thw=per_sample_thw[i],
            )
            phase2_inputs_list.append(dt_out)

        # LEFT-pad the per-sample inputs_embeds so that HF generate sees
        # a correctly aligned decoder input batch.
        B_trig = len(triggered_idx)
        max_L2 = max(d["inputs_embeds"].shape[1] for d in phase2_inputs_list)
        H_dim  = phase2_inputs_list[0]["inputs_embeds"].shape[-1]
        emb_dtype = phase2_inputs_list[0]["inputs_embeds"].dtype
        padded_embeds = torch.zeros(B_trig, max_L2, H_dim,
                                    device=device, dtype=emb_dtype)
        padded_mask   = torch.zeros(B_trig, max_L2,
                                    device=device, dtype=torch.long)
        padded_ids = torch.full((B_trig, max_L2), pad_token_id, device=device, dtype=torch.long)
        thw_chunks = []
        for j, d in enumerate(phase2_inputs_list):
            L_j = d["inputs_embeds"].shape[1]
            padded_embeds[j, -L_j:] = d["inputs_embeds"][0]
            padded_mask  [j, -L_j:] = d["attention_mask"][0]
            padded_ids[j, -L_j:] = d["input_ids"][0]
            thw_chunks.append(d["image_grid_thw"])
        combined_thw = torch.cat(thw_chunks, dim=0)  # [2*B_trig, 3]

        phase2_out = self.generate_vanilla(
            input_ids=padded_ids,
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            image_grid_thw=combined_thw,
            max_new_tokens=max_new_tokens_phase2,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **gen_kwargs,
        )
        phase2_out = phase2_out[:, max_L2:]
        # phase2_out: new tokens only after removing the structural prefix.

        # Attach per-sample phase-2 tokens (up to first eos / pad).
        for j, i in enumerate(triggered_idx):
            row = phase2_out[j].tolist()
            cut = len(row)
            for k, tok in enumerate(row):
                if tok in eos_token_id:
                    cut = k + 1  # keep the eos token
                    break
                if tok == pad_token_id:
                    cut = k
                    break
            new_toks = row[:cut]
            if new_toks:
                extra = torch.tensor(new_toks, dtype=trajectories[i].dtype,
                                     device=device)
                trajectories[i] = torch.cat([trajectories[i], extra], dim=0)

    # ---------------- Reassemble into HF-standard layout -------------------
    # HF ``generate`` returns ``[B, S + max_new]`` where the first ``S``
    # columns are **exactly** the input_ids as the caller passed them in
    # (keeping whatever padding layout they used), and the last ``max_new``
    # columns are the newly generated tokens (right-padded).  Downstream
    # consumers (lmms-eval) slice with ``cont[i][len(in_ids[i]):]`` and rely
    # on this layout, so we must preserve it.
    new_tokens_list: list[torch.Tensor] = []
    for i in range(B):
        real_prompt_len = int(attention_mask[i].sum().item())
        # trajectories[i] = [real_prompt(real_prompt_len), phase1_new, phase2_new]
        new_tokens_list.append(trajectories[i][real_prompt_len:])

    max_new = max((t.shape[0] for t in new_tokens_list), default=0)
    out = torch.full((B, S + max_new), pad_token_id,
                     dtype=input_ids.dtype, device=device)
    out[:, :S] = input_ids   # preserve the caller's original padding layout
    for i in range(B):
        n = new_tokens_list[i].shape[0]
        if n > 0:
            out[i, S: S + n] = new_tokens_list[i]
    return out


# ---------------------------------------------------------------------------
# public replacement for generate()
# ---------------------------------------------------------------------------

@torch.no_grad()
def multistage_generate(self: Qwen2_5_VLForConditionalGeneration,
                        *args, **kwargs):
    """Drop-in replacement for ``generate`` that runs the two-phase paradigm
    automatically when appropriate.

    Extra kwargs (all optional)
    ---------------------------
    multistage : bool | None
        ``None`` (default) -> auto.  ``True`` -> force two-phase (error if
        prereqs missing).  ``False`` -> always vanilla.
    max_new_tokens_phase1 / max_new_tokens_phase2 : int
        Caps for phase-1/phase-2. Default to ``max_new_tokens`` (or 512).
    generate_type : str
        Forwarded to ``pass_through_visual_detransformer``
        (default ``"Scaler"``).
    """
    # ---- pop our own switches ----
    multistage          = kwargs.pop("multistage", None)
    max_new_phase1      = kwargs.pop("max_new_tokens_phase1", None)
    max_new_phase2      = kwargs.pop("max_new_tokens_phase2", None)
    generate_type       = kwargs.pop("generate_type", "Scaler")

    # ---- inputs ----
    input_ids        = kwargs.get("input_ids", args[0] if args else None)
    pixel_values     = kwargs.get("pixel_values", None)
    image_grid_thw   = kwargs.get("image_grid_thw", None)
    inputs_embeds    = kwargs.get("inputs_embeds", None)
    attention_mask   = kwargs.get("attention_mask", None)

    auxiliary_start_id = _get_auxiliary_start_id(self)

    can_multistage = (
        auxiliary_start_id is not None
        and inputs_embeds is None
        and pixel_values is not None
        and image_grid_thw is not None
        and input_ids is not None
        and hasattr(self, "pass_through_visual_detransformer")
    )

    if multistage is None:
        do_multistage = can_multistage
    elif multistage is True:
        if not can_multistage:
            raise ValueError(
                "multistage=True requested but prerequisites missing: "
                f"auxiliary_start_id={auxiliary_start_id!r}, "
                f"input_ids given={input_ids is not None}, "
                f"pixel_values given={pixel_values is not None}, "
                f"image_grid_thw given={image_grid_thw is not None}, "
                f"inputs_embeds given={inputs_embeds is not None} (must be None), "
                f"has pass_through_visual_detransformer={hasattr(self, 'pass_through_visual_detransformer')}."
            )
        do_multistage = True
    else:
        do_multistage = False

    if not do_multistage:
        return self.generate_vanilla(*args, **kwargs)

    # ---- resolve caps / eos / pad ----
    mnt_fallback = kwargs.get("max_new_tokens", 512)
    if max_new_phase1 is None:
        max_new_phase1 = mnt_fallback
    if max_new_phase2 is None:
        max_new_phase2 = mnt_fallback
    kwargs.pop("max_new_tokens", None)

    eos_list = _as_list_eos(
        kwargs.pop("eos_token_id", None) or self.generation_config.eos_token_id
    )
    pad_id = kwargs.pop("pad_token_id", None)
    if pad_id is None:
        pad_id = self.generation_config.pad_token_id
    if pad_id is None:
        pad_id = eos_list[0] if eos_list else 0

    # Strip inputs from kwargs; we pass them explicitly.
    kwargs.pop("input_ids", None)
    kwargs.pop("attention_mask", None)
    kwargs.pop("pixel_values", None)
    kwargs.pop("image_grid_thw", None)
    kwargs.pop("inputs_embeds", None)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    B = input_ids.shape[0]
    if len(image_grid_thw) != B:
        raise ValueError("SFT two-phase inference requires exactly one image per sample")
    if int(kwargs.get("num_beams", 1)) != 1:
        raise ValueError("Two-phase inference currently supports num_beams=1 only")
    self.model.rope_deltas = None

    if B == 1:
        return _single_sample_two_phase(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            auxiliary_start_id=auxiliary_start_id,
            generate_type=generate_type,
            max_new_tokens_phase1=max_new_phase1,
            max_new_tokens_phase2=max_new_phase2,
            eos_token_id=eos_list,
            pad_token_id=pad_id,
            **kwargs,
        )

    return _batched_two_phase(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        auxiliary_start_id=auxiliary_start_id,
        generate_type=generate_type,
        max_new_tokens_phase1=max_new_phase1,
        max_new_tokens_phase2=max_new_phase2,
        eos_token_id=eos_list,
        pad_token_id=pad_id,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------

_ORIG_GENERATE_ATTR = "generate_vanilla"


def patch_multistage_generate(force: bool = False) -> None:
    """Rebind ``Qwen2_5_VLForConditionalGeneration.generate``.

    After this:
        * ``model.generate_vanilla(...)`` -> original HF ``generate``.
        * ``model.generate(...)``         -> two-phase wrapper.

    Idempotent unless ``force=True``.
    """
    cls = Qwen2_5_VLForConditionalGeneration
    already_patched = getattr(cls.generate, "_is_multistage_patch", False)
    if already_patched and not force:
        return
    if not hasattr(cls, _ORIG_GENERATE_ATTR) or force:
        cls.generate_vanilla = cls.generate  # inherited from GenerationMixin

    multistage_generate._is_multistage_patch = True  # type: ignore[attr-defined]
    cls.generate = multistage_generate


__all__ = ["multistage_generate", "patch_multistage_generate"]
