# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Perform image traversals using a trained MERU or CLIP model, and a pool of
text (and their encoded text representations).
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from PIL import Image
from torchvision import transforms as T

import pandas as pd  # <-- NEW

from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--image-path", help="Path to an image (.jpg) for perfoming traversal.")
_AA("--steps", type=int, default=50, help="Number of traversal steps.")

# NEW: CSV with prompts + nudity filtering
_AA("--csv-path", type=str, help="Path to CSV file containing target prompts.")
_AA(
    "--nudity",
    action="store_true",
    help="If set, filter CSV by nudity_percentage > 0 (if column exists).",
)
_AA(
    "--prompt-column",
    type=str,
    default="target_prompt",
    help="Name of column in CSV containing prompts.",
)


def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type.
    """

    # Linear interpolation between root and image features. For MERU, this happens
    # in the tangent space of the origin.
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid (for MERU), or L2 normalize (for CLIP).
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)


def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    #if False:
    if isinstance(model, MERU):
        scores = L.pairwise_inner(image_feats, text_feats, model.curv.exp())

        # For MERU, exclude text embeddings that do not entail the given image.
        _aper = L.half_aperture(text_feats, model.curv.exp())
        _oxy_angle = L.oxy_angle(
            text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        )
        entailment_energy = _oxy_angle - _aper[..., None]

        # Root entails everything.
        if has_root:
            entailment_energy[-1, ...] = 0
        #print(entailment_energy)
        # Set a large negative score if text does not entail image.
        scores[entailment_energy.T > 1e-2] = -1e12
        return scores
    else:
        # model is not needed here.
        return image_feats @ text_feats.T


@torch.inference_mode()
def get_text_feats(model: MERU | CLIPBaseline) -> tuple[list[str], torch.Tensor]:
    # Get all captions, nouns, and adjectives collected from pexels.com website
    pexels_text = json.load(open("assets/nsfw_pexels.json"))

    # Use very simple prompts for noun and adjective tags.
    tokenizer = Tokenizer()
    NOUN_PROMPT = "{}"
    ADJ_PROMPT = "this is {}."

    all_text_feats = []

    # Tokenize and encode captions.
    #caption_tokens = tokenizer(pexels_text["captions"])
    #all_text_feats.append(model.encode_text(caption_tokens, project=True))

    # Tokenize and encode prompts filled with tags.
    noun_prompt_tokens = tokenizer(
        [NOUN_PROMPT.format(tag) for tag in pexels_text["nouns"]]
    )
    all_text_feats.append(model.encode_text(noun_prompt_tokens, project=True))

    adj_prompt_tokens = tokenizer(
        [ADJ_PROMPT.format(tag) for tag in pexels_text["adjectives"]]
    )
    all_text_feats.append(model.encode_text(adj_prompt_tokens, project=True))

    all_text_feats = torch.cat(all_text_feats, dim=0)
    all_pexels_text = [
        *pexels_text["captions"],
        *pexels_text["nouns"],
        *pexels_text["adjectives"],
    ]
    return all_pexels_text, all_text_feats


def load_and_filter_prompts(args: argparse.Namespace) -> Tuple[List[str], Optional[List[Optional[float]]]]:
    """
    Load prompts from CSV and optionally filter by nudity_percentage.
    Returns:
        prompts: list of prompt strings
        nudity_percentages: list of floats (or None if unavailable), aligned with prompts,
                            or None if no CSV was provided.
    """
    if args.csv_path is None:
        # No CSV: single dummy "prompt" so main loop still runs once.
        return ["__single_run__"], None

    df = pd.read_csv(args.csv_path, index_col=0)

    # If a nudity_percentage column exists, make it numeric first
    if "nudity_percentage" in df.columns:
        df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")

        # If requested, filter and sort by nudity_percentage
        if args.nudity:
            # keep rows with nudity_percentage > 0
            df = df[df["nudity_percentage"].gt(0)]
            # sort descending
            df = df.sort_values(by="nudity_percentage", ascending=False)

    # Now extract prompts
    if args.prompt_column not in df.columns:
        raise ValueError(
            f"Prompt column '{args.prompt_column}' not found in CSV "
            f"(available: {list(df.columns)})"
        )

    prompts = df[args.prompt_column].astype(str).dropna().tolist()

    # Align nudity percentages with those prompts (if column exists)
    if "nudity_percentage" in df.columns:
        nudity_series = df.loc[df[args.prompt_column].astype(str).notna(), "nudity_percentage"]
        nudity_percentages_raw = nudity_series.tolist()

        # Replace NaN with None for convenience
        nudity_percentages: List[Optional[float]] = [
            (None if (isinstance(x, float) and math.isnan(x)) else x)
            for x in nudity_percentages_raw
        ]
    else:
        nudity_percentages = [None] * len(prompts)

    return prompts, nudity_percentages

@torch.inference_mode()
def main(_A: argparse.Namespace):
    # Get device
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the model using training config and load pre-trained weights.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)

    if isinstance(model, MERU):
        root_feat = torch.zeros(_C_TRAIN.model.embed_dim, device=device)
    else:
        # CLIP model checkpoint should have the 'root' embedding.
        root_feat = torch.load(_A.checkpoint_path, weights_only=False)["root"].to(
            device
        )

    # Compute text pool only once
    text_pool, text_feats_pool = get_text_feats(model)
    # Add [ROOT] to the pool of text feats.
    text_pool.append("[ROOT]")
    text_feats_pool = torch.cat([text_feats_pool, root_feat[None, ...]])

    prompts, nudity_percentages = load_and_filter_prompts(_A)
    # Load all prompts (and apply nudity filtering if requested)
    for i, prompt in enumerate(prompts):
        print("\n" + "=" * 80)

        # Get corresponding nudity percentage if available
        nudity_str = ""
        if nudity_percentages is not None:
            nudity = nudity_percentages[i]
            if nudity is not None:
                nudity_str = f" | nudity_percentage: {nudity:.2f}"

        if _A.csv_path is not None:
            print(f"[{i + 1}/{len(prompts)}] Target prompt: {prompt}{nudity_str}")
        else:
            print(f"[{i + 1}/{len(prompts)}] Single run (no CSV)")
        # --------------------------------------------------------------------
        print(f"Performing text traversals with source prompt: {prompt}...")
        # --------------------------------------------------------------------
        tokenizer = Tokenizer()

        text_tokens = tokenizer([prompt])
        text_feats = model.encode_text(text_tokens, project=True)[0]

        interp_feats = interpolate(model, text_feats, root_feat, _A.steps)
        nn1_scores = calc_scores(model, interp_feats, text_feats_pool, has_root=True)

        nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
        nn1_texts = [text_pool[_idx.item()] for _idx in _nn1_idxs]

        # De-duplicate retrieved texts (multiple points may have same NN) and print.
        print(f"Texts retrieved from [TEXT] -> [ROOT] traversal:")
        unique_nn1_texts = []
        for _text in nn1_texts:
            if _text not in unique_nn1_texts:
                unique_nn1_texts.append(_text)
                print(f"  - {_text}")

        # Optional: you could here log/save results together with `prompt`
        # if you want to associate traversal outputs with each target prompt.


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
