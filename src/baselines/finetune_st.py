"""
Fine-tune any sentence-transformers model with Triplet Loss
============================================================
Generic fork of finetune_minilm.py. Supports MiniLM (384-dim) and mpnet (768-dim)
out of the box; embedding dim is auto-detected from the loaded model.

All triplet, mining, bridge, U-computation, warmup, and early-stopping logic
preserved verbatim from the source. Adversarial / benchmark evaluation does NOT
live in this class — it belongs to the Phase D unified eval pipeline. This class
produces checkpoints and per-stage training logs only.

Telemetry additions over the source: per-epoch active_fraction (fraction of
mined triplets with positive loss) and grad_norm are now tracked and surfaced
through train_epoch / train_stage. The math is unchanged — only side-channel
counters were added.

Input: raw text descriptions.
Output: model state_dict checkpoints + per-stage training log.
"""

import json
import copy
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class STFineTuner:
    """Fine-tune a sentence-transformers model with triplet loss."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 seed=42, device="cpu"):
        self.model_name = model_name
        self.seed = seed
        self.device = device
        self._set_seed(seed)
        self.model = SentenceTransformer(model_name, device=device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        self.margin = 1.0
        self.tokenizer = self.model.tokenizer

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _encode_with_grad(self, texts):
        """Encode texts through the transformer WITH gradients for training."""
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=128, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model[0].auto_model(**encoded)
        # Mean pooling over token embeddings (matching ST default for both MiniLM and mpnet)
        attention_mask = encoded["attention_mask"]
        token_embs = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        sum_embs = (token_embs * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embs / sum_mask
        return F.normalize(pooled, p=2, dim=-1)

    def encode(self, texts):
        """Encode list of texts to (N, D) tensor, L2-normalized. No gradients."""
        embs = self.model.encode(texts, convert_to_tensor=True,
                                  show_progress_bar=False, device=self.device)
        return F.normalize(embs, p=2, dim=-1)

    def encode_single(self, text):
        """Encode single text to (D,) tensor, L2-normalized. No gradients."""
        return self.encode([text])[0]

    def _triplet_loss(self, anchor, positive, negative):
        d_pos = (anchor - positive).pow(2).sum()
        d_neg = (anchor - negative).pow(2).sum()
        return torch.clamp(d_pos - d_neg + self.margin, min=0.0)

    def _collect_concept_embeddings(self, config, concept_list, with_grad=False):
        """
        For each concept, embed all 7 variants.
        with_grad=True uses differentiable encoding for training.
        """
        result = {}
        for group_name, group_data in config["groups"].items():
            for obj_name, variants in group_data["objects"].items():
                if obj_name not in concept_list:
                    continue
                texts = []
                keys = []
                for var_name, var_text in variants.items():
                    if var_name == "ground":
                        continue
                    texts.append(var_text)
                    keys.append(var_name)
                if not texts:
                    continue
                if with_grad:
                    embs = self._encode_with_grad(texts)
                else:
                    embs = self.encode(texts)
                result[obj_name] = {k: embs[i] for i, k in enumerate(keys)}
        return result

    def _batch_hard_triplets(self, concept_embs, batch_concepts):
        """
        Batch-hard mining: for each concept's base embedding, find hardest positive
        (farthest same-concept variant) and hardest negative (closest different-concept base).
        """
        anchors, positives, negatives = [], [], []

        # Collect all base embeddings for negative mining
        all_bases = []
        all_base_concepts = []
        for c in batch_concepts:
            if c not in concept_embs or "base" not in concept_embs[c]:
                continue
            all_bases.append(concept_embs[c]["base"])
            all_base_concepts.append(c)

        if len(all_bases) < 2:
            return None, None, None

        base_stack = torch.stack(all_bases)  # (B, D)

        for c in batch_concepts:
            if c not in concept_embs or "base" not in concept_embs[c]:
                continue

            anchor = concept_embs[c]["base"]

            # Positive candidates: paraphrases + swaps
            pos_keys = ["paraphrase_1", "paraphrase_2", "attribute_swap_1", "attribute_swap_2"]
            pos_embs = [concept_embs[c][k] for k in pos_keys if k in concept_embs[c]]
            if not pos_embs:
                continue

            # Hardest positive: farthest same-concept variant
            pos_stack = torch.stack(pos_embs)
            pos_dists = (anchor.unsqueeze(0) - pos_stack).pow(2).sum(dim=-1)
            hardest_pos = pos_stack[pos_dists.argmax()]

            # Hardest negative: closest different-concept base
            neg_mask = torch.tensor([bc != c for bc in all_base_concepts], dtype=torch.bool)
            if neg_mask.sum() == 0:
                continue
            valid_negs = base_stack[neg_mask]
            neg_dists = (anchor.unsqueeze(0) - valid_negs).pow(2).sum(dim=-1)
            hardest_neg = valid_negs[neg_dists.argmin()]

            anchors.append(anchor)
            positives.append(hardest_pos)
            negatives.append(hardest_neg)

        if not anchors:
            return None, None, None

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def _standard_triplets(self, concept_embs, batch_concepts):
        """
        Standard triplets matching the brain's optimizer:
        anchor=base, pos=paraphrase/swap, neg=contradiction (cross-concept negatives via batch-hard).
        Returns (loss, count, active_fraction, n_total_triplets).
        """
        total_loss_s = torch.tensor(0.0, device=self.device)
        total_loss_c = torch.tensor(0.0, device=self.device)
        count = 0
        triplet_losses = []  # per-triplet, detached, for active_fraction telemetry

        # Collect all contradiction embeddings for cross-concept negatives
        all_contras = []
        all_contra_concepts = []
        for c in batch_concepts:
            if c not in concept_embs:
                continue
            for k in ["contradiction_1", "contradiction_2"]:
                if k in concept_embs[c]:
                    all_contras.append(concept_embs[c][k])
                    all_contra_concepts.append(c)

        if not all_contras:
            return torch.tensor(0.0, device=self.device), 0, 0.0, 0

        contra_stack = torch.stack(all_contras)

        for c in batch_concepts:
            if c not in concept_embs or "base" not in concept_embs[c]:
                continue

            anchor = concept_embs[c]["base"]

            # Cross-concept negative: closest contradiction from a DIFFERENT concept
            neg_mask = torch.tensor([cc != c for cc in all_contra_concepts], dtype=torch.bool)
            if neg_mask.sum() == 0:
                continue
            valid_negs = contra_stack[neg_mask]
            neg_dists = (anchor.unsqueeze(0) - valid_negs).pow(2).sum(dim=-1)
            hardest_neg = valid_negs[neg_dists.argmin()]

            n_s = 0
            n_c = 0

            # Semantic triplets: paraphrases
            for pk in ["paraphrase_1", "paraphrase_2"]:
                if pk in concept_embs[c]:
                    pos = concept_embs[c][pk]
                    tl = self._triplet_loss(anchor, pos, hardest_neg)
                    total_loss_s = total_loss_s + tl
                    triplet_losses.append(tl.detach())
                    n_s += 1

            # Causal triplets: attribute swaps
            for sk in ["attribute_swap_1", "attribute_swap_2"]:
                if sk in concept_embs[c]:
                    pos = concept_embs[c][sk]
                    tl = self._triplet_loss(anchor, pos, hardest_neg)
                    total_loss_c = total_loss_c + tl
                    triplet_losses.append(tl.detach())
                    n_c += 1

            if n_s > 0:
                total_loss_s = total_loss_s  # already accumulated
            if n_c > 0:
                total_loss_c = total_loss_c
            if n_s + n_c > 0:
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device), 0, 0.0, 0

        avg_s = total_loss_s / count
        avg_c = total_loss_c / count
        loss = 0.4 * avg_s + 0.4 * avg_c

        n_total = len(triplet_losses)
        if n_total > 0:
            active_fraction = float((torch.stack(triplet_losses) > 0).float().mean().item())
        else:
            active_fraction = 0.0

        return loss, count, active_fraction, n_total

    def _bridge_triplets(self, concept_embs, bridge_config):
        """
        Bridge triplets for Stage 3: bidirectional abstract ↔ physical.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_bridge = 0

        for abstract_name, bc in bridge_config.items():
            if abstract_name not in concept_embs or "base" not in concept_embs[abstract_name]:
                continue
            ground_name = bc["ground"]
            if ground_name not in concept_embs or "base" not in concept_embs[ground_name]:
                continue

            abstract_emb = concept_embs[abstract_name]["base"]
            ground_emb = concept_embs[ground_name]["base"]

            # Forward: abstract -> ground (pos) vs unrelated physical (neg)
            for ur_name in bc.get("unrelated_physicals", [])[:2]:
                if ur_name in concept_embs and "base" in concept_embs[ur_name]:
                    ur_emb = concept_embs[ur_name]["base"]
                    total_loss = total_loss + self._triplet_loss(abstract_emb, ground_emb, ur_emb)
                    n_bridge += 1

            # Reverse: ground -> abstract (pos) vs unrelated abstract (neg)
            for ur_name in bc.get("unrelated_abstracts", [])[:2]:
                if ur_name in concept_embs and "base" in concept_embs[ur_name]:
                    ur_emb = concept_embs[ur_name]["base"]
                    total_loss = total_loss + self._triplet_loss(ground_emb, abstract_emb, ur_emb)
                    n_bridge += 1

        return total_loss, n_bridge

    def _build_text_index(self, configs, concept_set):
        """Collect all variant texts for needed concepts. Returns {concept: {variant: text}}."""
        text_index = {}
        for cfg in configs:
            for gd in cfg["groups"].values():
                for obj_name, variants in gd["objects"].items():
                    if obj_name not in concept_set or obj_name in text_index:
                        continue
                    text_index[obj_name] = {
                        k: v for k, v in variants.items() if k != "ground"
                    }
        return text_index

    def _batch_encode_concepts(self, text_index, concept_set, with_grad=False):
        """
        Encode all variants for all concepts in ONE forward pass.
        Returns {concept: {variant: embedding_tensor}}.
        """
        all_texts = []
        index_map = []  # (concept, variant, position_in_batch)
        for c in concept_set:
            if c not in text_index:
                continue
            for var_name, var_text in text_index[c].items():
                index_map.append((c, var_name))
                all_texts.append(var_text)

        if not all_texts:
            return {}

        # Single forward pass for all texts
        if with_grad:
            all_embs = self._encode_with_grad(all_texts)
        else:
            all_embs = self.encode(all_texts)

        # Map back to concept/variant structure
        result = {}
        for i, (c, var_name) in enumerate(index_map):
            if c not in result:
                result[c] = {}
            result[c][var_name] = all_embs[i]

        return result

    def train_epoch(self, optimizer, configs, concept_list, batch_size=16,
                    bridge_config=None, all_configs=None):
        """One training epoch: sample batch, encode all in one pass, triplet loss, backprop.

        Returns (loss_value, stats_dict) where stats_dict carries:
            loss, active_fraction, grad_norm, n_triplets, n_bridge.
        """
        self.model.train()

        batch = random.sample(concept_list, min(batch_size, len(concept_list)))

        # Determine all concepts needed (batch + bridge targets)
        all_concepts_needed = set(batch)
        if bridge_config:
            for abstract_name in batch:
                if abstract_name in bridge_config:
                    bc = bridge_config[abstract_name]
                    all_concepts_needed.add(bc["ground"])
                    all_concepts_needed.update(bc.get("unrelated_physicals", [])[:2])
                    all_concepts_needed.update(bc.get("unrelated_abstracts", [])[:2])

        # Build text index from all configs
        all_cfgs = configs + (all_configs or [])
        text_index = self._build_text_index(all_cfgs, all_concepts_needed)

        # Single batched forward pass with gradients
        concept_embs = self._batch_encode_concepts(text_index, all_concepts_needed, with_grad=True)

        # Standard triplet loss
        loss, count, active_fraction, n_triplets = self._standard_triplets(concept_embs, batch)

        # Bridge triplets (only for batch concepts that have bridge config)
        bridge_loss = torch.tensor(0.0, device=self.device)
        n_bridge = 0
        if bridge_config:
            batch_bridge = {k: v for k, v in bridge_config.items() if k in batch}
            bridge_loss, n_bridge = self._bridge_triplets(concept_embs, batch_bridge)

        if count == 0:
            return 0.0, {
                "loss": 0.0, "standard_loss": 0.0, "bridge_loss": 0.0,
                "active_fraction": 0.0, "grad_norm": 0.0,
                "n_triplets": 0, "n_bridge": 0,
            }

        standard_loss_val = float(loss.item())  # 0.4*avg_s + 0.4*avg_c, pre-bridge
        bridge_loss_val = 0.0
        total_loss = loss
        if n_bridge > 0:
            avg_bridge = bridge_loss / n_bridge
            bridge_loss_val = float(avg_bridge.item())  # unweighted per-triplet avg
            total_loss = total_loss + 0.2 * avg_bridge  # 0.2 weight matches the brain

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        gn_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

        return total_loss.item(), {
            "loss": total_loss.item(),
            "standard_loss": standard_loss_val,
            "bridge_loss": bridge_loss_val,
            "active_fraction": active_fraction,
            "grad_norm": gn_val,
            "n_triplets": n_triplets,
            "n_bridge": n_bridge,
        }

    def compute_u_text(self, config, concept_list):
        """
        Compute U scores using text embeddings (no graph, no brain).
        Same U formula: U = 0.4*U_s + 0.4*U_c + 0.2*U_n.
        """
        self.model.eval()
        scores = {}

        # Batch encode all concepts at once
        text_index = self._build_text_index([config], set(concept_list))
        with torch.no_grad():
            all_embs = self._batch_encode_concepts(text_index, set(concept_list), with_grad=False)

        for obj_name in concept_list:
            if obj_name not in all_embs:
                continue
            embs = all_embs[obj_name]
            if "base" not in embs:
                continue

            # U_s: base + paraphrases consistency
            bp_embs = [embs[k] for k in ["base", "paraphrase_1", "paraphrase_2"] if k in embs]
            if len(bp_embs) >= 2:
                sims = []
                for i in range(len(bp_embs)):
                    for j in range(i + 1, len(bp_embs)):
                        sim = F.cosine_similarity(bp_embs[i].unsqueeze(0), bp_embs[j].unsqueeze(0))
                        sims.append((sim.item() + 1) / 2)
                u_s = np.mean(sims)
            else:
                u_s = 0.0

            # U_c: base vs swaps
            sw_embs = [embs[k] for k in ["attribute_swap_1", "attribute_swap_2"] if k in embs]
            if sw_embs:
                sims = []
                base_emb = embs["base"]
                for se in sw_embs:
                    sim = F.cosine_similarity(base_emb.unsqueeze(0), se.unsqueeze(0))
                    sims.append((sim.item() + 1) / 2)
                u_c = np.mean(sims)
            else:
                u_c = 0.0

            # U_n: base vs contradictions (1 - sim)
            co_embs = [embs[k] for k in ["contradiction_1", "contradiction_2"] if k in embs]
            if co_embs:
                sims = []
                base_emb = embs["base"]
                for ce in co_embs:
                    sim = F.cosine_similarity(base_emb.unsqueeze(0), ce.unsqueeze(0))
                    sims.append((sim.item() + 1) / 2)
                u_n = 1.0 - np.mean(sims)
            else:
                u_n = 0.0

            u_total = 0.4 * u_s + 0.4 * u_c + 0.2 * u_n
            scores[obj_name] = {"U_s": u_s, "U_c": u_c, "U_n": u_n, "U_total": u_total}

        return scores

    def train_stage(self, stage_name, configs, concept_list, epochs=500, lr=2e-5,
                    warmup_epochs=10, batch_size=16, bridge_config=None,
                    all_configs=None, heldout_config=None, heldout_concepts=None,
                    ckpt_dir=None):
        """
        Train one curriculum stage with early stopping.

        Returns (best_epoch, best_heldout_u, log) — log is a list of per-eval-step dicts
        with epoch, loss, active_fraction, grad_norm, n_triplets, n_bridge, lr, train_U, heldout_U.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Warmup scheduler
        warmup_lr_start = lr / 100

        best_heldout_u = -float("inf")
        best_state = None
        epochs_without_improvement = 0
        best_epoch = 0
        log = []

        print(f"\n  Training {stage_name}: {epochs} epochs, lr={lr}, "
              f"{len(concept_list)} concepts, batch={batch_size}")

        for epoch in range(epochs):
            # Linear warmup
            if epoch < warmup_epochs:
                current_lr = warmup_lr_start + (lr - warmup_lr_start) * (epoch / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            loss_val, stats = self.train_epoch(
                optimizer, configs, concept_list, batch_size=batch_size,
                bridge_config=bridge_config, all_configs=all_configs,
            )

            # Evaluate every 10 epochs
            if epoch % 10 == 0:
                # Training U
                train_scores = {}
                for cfg in configs:
                    train_scores.update(self.compute_u_text(cfg, concept_list))
                train_u = np.mean([s["U_total"] for s in train_scores.values()]) if train_scores else 0

                # Held-out U
                ho_u = 0.0
                if heldout_config and heldout_concepts:
                    ho_scores = self.compute_u_text(heldout_config, heldout_concepts)
                    ho_u = np.mean([s["U_total"] for s in ho_scores.values()]) if ho_scores else 0

                log.append({
                    "epoch": epoch,
                    "loss": round(stats["loss"], 5),
                    "standard_loss": round(stats["standard_loss"], 5),
                    "bridge_loss": round(stats["bridge_loss"], 5),
                    "active_fraction": round(stats["active_fraction"], 4),
                    "grad_norm": round(stats["grad_norm"], 4),
                    "n_triplets": stats["n_triplets"],
                    "n_bridge": stats["n_bridge"],
                    "lr": round(optimizer.param_groups[0]["lr"], 7),
                    "train_U": round(float(train_u), 4),
                    "heldout_U": round(float(ho_u), 4),
                })

                if epoch % 50 == 0:
                    print(f"    Epoch {epoch:4d}: loss={loss_val:.4f}  "
                          f"active={stats['active_fraction']:.2f}  "
                          f"grad={stats['grad_norm']:.2f}  "
                          f"train_U={train_u:.4f}  heldout_U={ho_u:.4f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")

                # Early stopping check on held-out U
                check_val = ho_u if heldout_concepts else train_u
                if check_val > best_heldout_u:
                    best_heldout_u = check_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10

                if epochs_without_improvement >= 100:
                    print(f"    Early stopping at epoch {epoch}: no improvement for 100 epochs")
                    print(f"    Best checkpoint at epoch {best_epoch}, U={best_heldout_u:.4f}")
                    break

                # Collapse detection: only on true loss explosion
                if loss_val > 2.0:
                    print(f"    WARNING: Loss spike (loss={loss_val:.4f}). "
                          f"Reducing lr by half and reloading best checkpoint")
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    lr = lr / 2
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    epochs_without_improvement = 0

        # Load best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save checkpoint
        if ckpt_dir:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), Path(ckpt_dir) / "best.pt")

        print(f"    {stage_name} complete: best epoch={best_epoch}, best_U={best_heldout_u:.4f}")
        return best_epoch, best_heldout_u, log

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
