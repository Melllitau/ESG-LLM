#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised QA fine-tuning (SFT) com LoRA para modelos Causal LM
a partir de um dataset JSON/JSONL contendo, por exemplo:

[
  {
    "split": "train",
    "question": "What is ESG?",
    "answer": "ESG stands for Environmental, Social, and Governance."
  },
  ...
]

ou JSON com {"data": [...]} / dict de registros.

Regras:
- usa apenas exemplos com split == "train"
- espera chaves de pergunta e resposta (default: question, answer)
- faz treino estilo QA/instruction
- calcula loss apenas sobre a resposta

Exemplo:

export HF_TOKEN="   "  # opcional, para push no Hugging Face Hub
python train_qa.py \
  --json_file pira/all_splits.json \
  --split_key split \
  --train_split_value train \
  --question_column question_pt_origin \
  --answer_column answer_pt_origin \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --repo_adapters Qwen3-4B-it-pira-ep3-QA-qairm-ptbr \
  --repo_merged Qwen3-4B-it-pira-ep3-QA-qairm-ptbr \
  --wandb_project QA-IRM-Train \
  --wandb_run_name Qwen3-4B-it-pira-QA-ep3-ptbr \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_length 1024 \
  --eco2ai_enable \
  --eco2ai_project Qwen3-4B-it-pira-QA-ptbr \
  --eco2ai_experiment Qwen3-4B-it-pira-ep3-QA-ptbr \
  --eco2ai_file emissions_Qwen3-4B-it-pira-ep3-QA-ptbr.csv
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel

# W&B opcional
try:
    import wandb
    wandb.login(key="a900edab5b90d3286c37d0e4e4f95ef4be5db3c8")
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


# ------------------ IO ------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in [".jsonl", ".jsonlines"]:
        data = _read_jsonl(p)
    else:
        data = _read_json(p)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "data" in data:
            d = data["data"]
            if isinstance(d, list):
                return d
            if isinstance(d, dict):
                return list(d.values())

        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    raise ValueError("Formato não reconhecido (use lista de objetos ou dict com 'data').")


def _norm(text: str) -> str:
    return " ".join((text or "").strip().split())


def collect_qa_records(
    records: List[Dict[str, Any]],
    question_column: str,
    answer_column: str,
    split_key: str,
    train_split_value: str,
    deduplicate: bool = True,
) -> List[Dict[str, str]]:
    cleaned = []
    seen = set()
    total = 0
    kept = 0

    for r in records:
        if not isinstance(r, dict):
            continue

        total += 1

        if split_key not in r:
            continue
        if str(r[split_key]).strip() != train_split_value:
            continue

        q = r.get(question_column, None)
        a = r.get(answer_column, None)

        if not isinstance(q, str) or not isinstance(a, str):
            continue

        qn = _norm(q)
        an = _norm(a)

        if not qn or not an:
            continue

        key = (qn, an)
        if deduplicate and key in seen:
            continue

        seen.add(key)
        cleaned.append({
            "question": q.strip(),
            "answer": a.strip(),
        })
        kept += 1

    print(f"[data] total registros: {total}")
    print(f"[data] split='{train_split_value}' válidos: {kept}")

    if not cleaned:
        raise ValueError(
            f"Nenhum exemplo válido encontrado com {split_key} == '{train_split_value}' "
            f"e colunas '{question_column}' / '{answer_column}'."
        )

    return cleaned


def build_hf_dataset(examples: List[Dict[str, str]]) -> Dataset:
    return Dataset.from_list(examples)


# ------------------ Prompting ------------------
def build_prompt(question: str) -> str:
    return (
        "Answer the following question.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ------------------ Collator ------------------
class SupervisedQACollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for ex in features:
            prompt = build_prompt(ex["question"])
            answer = ex["answer"]

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

            # adiciona eos ao final da resposta
            if self.tokenizer.eos_token_id is not None:
                answer_ids = answer_ids + [self.tokenizer.eos_token_id]

            full_ids = prompt_ids + answer_ids

            # truncação pela direita
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]

            # labels: ignora prompt, aprende só a resposta
            labels = [-100] * len(prompt_ids) + answer_ids
            labels = labels[:self.max_length]

            attention_mask = [1] * len(full_ids)

            input_ids_list.append(full_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            },
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        max_seq_len = batch["input_ids"].shape[1]

        padded_labels = []
        for labels in labels_list:
            pad_len = max_seq_len - len(labels)
            padded_labels.append(labels + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser("QA supervised fine-tuning com LoRA usando JSON/JSONL.")
    ap.add_argument("--json_file", type=str, required=True)

    ap.add_argument("--split_key", type=str, default="split")
    ap.add_argument("--train_split_value", type=str, default="train")

    ap.add_argument("--question_column", type=str, default="question")
    ap.add_argument("--answer_column", type=str, default="answer")

    ap.add_argument("--base_model", type=str, default="TucanoBR/Tucano-2b4")
    ap.add_argument("--repo_adapters", type=str, required=True)
    ap.add_argument("--repo_merged", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default="")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_strategy", type=str, default="epoch")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--deduplicate", action="store_true", help="Deduplica pares question-answer.")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # W&B
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)

    # eco2ai
    ap.add_argument("--eco2ai_enable", action="store_true")
    ap.add_argument("--eco2ai_project", type=str, default="qa-sft")
    ap.add_argument("--eco2ai_experiment", type=str, default=None)
    ap.add_argument("--eco2ai_file", type=str, default=None)

    args = ap.parse_args()

    # HF login
    tok_hf = args.hf_token or os.environ.get("HF_TOKEN", "")
    if tok_hf:
        login(token=tok_hf)

    torch.manual_seed(args.seed)

    # 1) Dados
    records = load_records(args.json_file)
    qa_examples = collect_qa_records(
        records=records,
        question_column=args.question_column,
        answer_column=args.answer_column,
        split_key=args.split_key,
        train_split_value=args.train_split_value,
        deduplicate=args.deduplicate,
    )
    ds = build_hf_dataset(qa_examples)

    # 2) eco2ai
    tracker = None
    if args.eco2ai_enable:
        try:
            import eco2ai
            eco_file = args.eco2ai_file or f"emissions_{args.repo_adapters.replace('/', '__')}.csv"
            exp_desc = args.eco2ai_experiment or args.repo_adapters
            tracker = eco2ai.Tracker(
                project_name=args.eco2ai_project,
                experiment_description=exp_desc,
                file_name=eco_file,
            )
            tracker.start()
            print(f"[eco2ai] tracking ON -> {eco_file}")
        except Exception as e:
            print(f"[eco2ai] aviso: não foi possível iniciar o tracker: {e}")

    try:
        # 3) Tokenizer e modelo
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if args.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False

        # 4) LoRA
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()

        collator = SupervisedQACollator(tokenizer=tok, max_length=args.max_length)

        # 5) W&B
        if args.wandb_project and _HAS_WANDB:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config={
                    "base_model": args.base_model,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "warmup_ratio": args.warmup_ratio,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "max_length": args.max_length,
                    "seed": args.seed,
                    "repo_adapters": args.repo_adapters,
                    "repo_merged": args.repo_merged,
                    "num_train_samples": len(qa_examples),
                    "split_key": args.split_key,
                    "train_split_value": args.train_split_value,
                    "question_column": args.question_column,
                    "answer_column": args.answer_column,
                },
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"

        # 6) Treino
        train_args = TrainingArguments(
            output_dir=args.repo_adapters,
            hub_model_id=args.repo_adapters,
            push_to_hub=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            bf16=True,
            optim="adamw_torch",
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            report_to=[] if os.environ.get("WANDB_DISABLED") == "true" else ["wandb"],
            seed=args.seed,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=ds,
            data_collator=collator,
        )

        trainer.train()

        if tracker is not None:
            try:
                tracker.stop()
                eco_file = tracker.file_name if hasattr(tracker, "file_name") else (
                    args.eco2ai_file or "emissions.csv"
                )
                last_row = None
                if os.path.exists(eco_file):
                    with open(eco_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]

                if last_row and args.wandb_project and _HAS_WANDB:
                    payload = {
                        "eco2ai_duration_s": float(last_row.get("duration(s)", "nan")),
                        "eco2ai_power_kWh": float(last_row.get("power_consumption(kWTh)", "nan")),
                        "eco2ai_co2_kg": float(last_row.get("CO2_emissions(kg)", "nan")),
                        "eco2ai_country": last_row.get("country", "unknown"),
                        "eco2ai_gpu": last_row.get("GPU_name", "unknown"),
                        "eco2ai_cpu": last_row.get("CPU_name", "unknown"),
                        "eco2ai_log_file": eco_file,
                    }
                    wandb.log(payload)
                    print(f"[eco2ai] métricas enviadas ao W&B: {payload}")
                else:
                    print("[eco2ai] tracking OFF ou sem arquivo para log.")
            except Exception as e:
                print(f"[eco2ai] aviso ao finalizar: {e}")

        trainer.push_to_hub()
        tok.push_to_hub(args.repo_adapters)

        # 7) Merge LoRA
        merged_base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        peft_model = PeftModel.from_pretrained(merged_base, args.repo_adapters, device_map="auto")
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(args.repo_merged, safe_serialization=True, push_to_hub=True)
        tok.push_to_hub(args.repo_merged)

        print(f"\n✅ Adapters (LoRA): https://huggingface.co/{args.repo_adapters}")
        print(f"✅ Merged model:     https://huggingface.co/{args.repo_merged}")

    finally:
        print(">> Finalizando...")


if __name__ == "__main__":
    main()