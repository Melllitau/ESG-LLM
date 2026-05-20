#!/usr/bin/env python3
"""
Continued pretraining (CLM) com LoRA
a partir de um dataset JSON/JSONL, usando a coluna especificada
(padrão: 'context'), com deduplicação de textos antes do treino.

Entrada aceita:
- JSON com lista de objetos: [{"context": "...", ...}, ...]
- JSON com dict contendo chave 'data' -> lista ou dict de registros
- JSONL: um objeto por linha

Exemplos:
  
BASE 3 epochs
  python train_context.py \
  --json_file esg_full_splited_set70-10-20.json \
  --text_column context \
  --repo_adapters Qwen3-4B-base-lora-esg-ep3 \
  --repo_merged  Qwen3-4B-base-esg-ep3 \
  --wandb_project ESG-Train \
  --wandb_run_name Qwen3-4B-base-esg-CT-ep3 \
  --epochs 3 --base_model Qwen/Qwen3-4B-Base --batch_size 4 \
  --eco2ai_enable \
  --eco2ai_project Qwen3-4B-esg-CT \
  --eco2ai_experiment Qwen3-4B-base-esg-ep3 \
  --eco2ai_file emissions_Qwen3-4B-base-esg-ep3.csv

# JSONL também é aceito:
python train_context.py \
  --json_file dataset.jsonl \
  --text_column context \
  --epochs 10 --lr 1e-4 --batch_size 2 --grad_accum 16
"""

import os, json, csv, argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

# W&B opcional
try:
    import wandb
    _HAS_WANDB = True
    wandb.login(key="a900edab5b90d3286c37d0e4e4f95ef4be5db3c8")
except Exception:
    _HAS_WANDB = False

# ------------------ IO & Dedup ------------------
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

def collect_unique_texts(records: List[Dict[str, Any]], text_column: str) -> List[str]:
    seen, cleaned = set(), []
    total = 0
    for r in records:
        if not isinstance(r, dict): continue
        if text_column not in r:     continue
        val = r[text_column]
        if not isinstance(val, str): continue
        total += 1
        n = _norm(val)
        if n and n not in seen:
            seen.add(n)
            cleaned.append(val.strip())
    print(f"[dedup] '{text_column}': {total} → {len(cleaned)} únicos")
    if not cleaned:
        raise ValueError(f"Nenhum texto válido em '{text_column}'.")
    return cleaned

def build_hf_dataset(texts: List[str]) -> Dataset:
    return Dataset.from_list([{"text": t} for t in texts])

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser("Continued pretraining com LoRA usando coluna (default: context) + eco2ai.")
    ap.add_argument("--json_file",     type=str, required=True)
    ap.add_argument("--text_column",   type=str, default="context")
    ap.add_argument("--base_model",    type=str, default="TucanoBR/Tucano-2b4")
    ap.add_argument("--repo_adapters", type=str, required=True)
    ap.add_argument("--repo_merged",   type=str, required=True)
    ap.add_argument("--hf_token",      type=str, default="")
    ap.add_argument("--epochs",        type=int,   default=50)
    ap.add_argument("--lr",            type=float, default=2e-4)
    ap.add_argument("--warmup_ratio",  type=float, default=0.05)
    ap.add_argument("--batch_size",    type=int,   default=2)
    ap.add_argument("--grad_accum",    type=int,   default=8)
    ap.add_argument("--max_length",    type=int,   default=1000)
    ap.add_argument("--logging_steps", type=int,   default=20)
    ap.add_argument("--save_strategy", type=str,   default="epoch")
    ap.add_argument("--seed",          type=int,   default=13)
    # W&B
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_run_name",type=str, default=None)
    ap.add_argument("--wandb_entity",  type=str, default=None)
    # eco2ai
    ap.add_argument("--eco2ai_enable", action="store_true", help="Ativa o eco2ai Tracker()")
    ap.add_argument("--eco2ai_project", type=str, default="tucano-context-ft")
    ap.add_argument("--eco2ai_experiment", type=str, default=None,
                    help="Descrição do experimento; default: <repo_adapters>")
    ap.add_argument("--eco2ai_file", type=str, default=None,
                    help="CSV de saída do eco2ai; default: emissions_<repo>.csv")
    args = ap.parse_args()

    # HF login
    tok_hf = args.hf_token or os.environ.get("HF_TOKEN", "")
    if tok_hf:
        login(token=tok_hf)

    torch.manual_seed(args.seed)

    # 1) Dados & dedup
    records = load_records(args.json_file)
    texts   = collect_unique_texts(records, args.text_column)
    ds      = build_hf_dataset(texts)

    # 2) eco2ai (iniciar ANTES do uso pesado de CPU/GPU)
    tracker = None
    if args.eco2ai_enable:
        try:
            import eco2ai  # pip install eco2ai
            eco_file = args.eco2ai_file or f"emissions_{args.repo_adapters.replace('/', '__')}.csv"
            exp_desc = args.eco2ai_experiment or args.repo_adapters
            tracker  = eco2ai.Tracker(
                project_name=args.eco2ai_project,
                experiment_description=exp_desc,
                file_name=eco_file
            )
            tracker.start()
            print(f"[eco2ai] tracking ON → {eco_file}")
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
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        # 4) LoRA
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            task_type="CAUSAL_LM", bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        )
        model = get_peft_model(base, lora_cfg)

        # 5) Tokenização
        def tok_fn(batch):
            return tok(batch["text"], truncation=True, max_length=args.max_length, padding=False)
        ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

        # 6) W&B (opcional)
        if args.wandb_project and _HAS_WANDB:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config={
                    "text_column": args.text_column,
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
                    "num_samples_dedup": len(texts),
                },
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"

        # 7) Treino
        train_args = TrainingArguments(
            output_dir                  = args.repo_adapters,
            hub_model_id                = args.repo_adapters,
            push_to_hub                 = True,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            num_train_epochs            = args.epochs,
            learning_rate               = args.lr,
            lr_scheduler_type           = "cosine",
            warmup_ratio                = args.warmup_ratio,
            bf16                        = True,
            optim                       = "adamw_torch",
            logging_steps               = args.logging_steps,
            save_strategy               = args.save_strategy,
            report_to                   = ([] if os.environ.get("WANDB_DISABLED") == "true" else ["wandb"]),
            seed                        = args.seed,
        )

        trainer = Trainer(model=model, args=train_args, train_dataset=ds_tok, data_collator=collator)
        trainer.train()
        
        if tracker is not None:
            try:
                tracker.stop()
                eco_file = tracker.file_name if hasattr(tracker, "file_name") else (args.eco2ai_file or "emissions.csv")
                last_row = None
                if os.path.exists(eco_file):
                    with open(eco_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                if last_row and args.wandb_project and _HAS_WANDB:
                    # Colunas padrão do eco2ai conforme PyPI
                    payload = {
                        "eco2ai_duration_s": float(last_row.get("duration(s)", "nan")),
                        "eco2ai_power_kWh":  float(last_row.get("power_consumption(kWTh)", "nan")),
                        "eco2ai_co2_kg":     float(last_row.get("CO2_emissions(kg)", "nan")),
                        "eco2ai_country":    last_row.get("country", "unknown"),
                        "eco2ai_gpu":        last_row.get("GPU_name", "unknown"),
                        "eco2ai_cpu":        last_row.get("CPU_name", "unknown"),
                        "eco2ai_log_file":   eco_file,
                    }
                    wandb.log(payload)
                    print(f"[eco2ai] métricas enviadas ao W&B: {payload}")
                else:
                    print("[eco2ai] tracking OFF ou sem arquivo para log.")
            except Exception as e:
                print(f"[eco2ai] aviso ao finalizar: {e}")
        
        trainer.push_to_hub()
        tok.push_to_hub(args.repo_adapters)

        # 8) Mesclar LoRA → base e publicar
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