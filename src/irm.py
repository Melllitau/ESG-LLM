"""
Instruction Residual Method (Hugging Face)

Uso (exemplo):
export HF_TOKEN="   "  # opcional, para push no Hugging Face Hub
  python irm.py \
    --base-old Qwen/Qwen3-4B-Base \
    --instruct-old Qwen/Qwen3-4B-Thinking-2507 \
    --base-new g-assismoraes/Qwen3-4B-base-esg-ep3 \
    --out-repo g-assismoraes/Qwen3-4B-irm-esg-ep3 \
    --trust-remote-code
"""
import os, argparse, gc, sys
from pathlib import Path
from typing import Dict, Optional
import torch
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    AutoConfig,
)

def parse_args():
    p = argparse.ArgumentParser(description="Portar 'instruct' via Instruction Residual (delta de pesos).")
    p.add_argument("--hf_token",      type=str, default="", help="Token do HF (opcional).")
    p.add_argument("--base-old", required=True, help="Repo ID ou caminho local do Base (geração d1).")
    p.add_argument("--instruct-old", required=True, help="Repo ID ou caminho local do Instruct (mesma geração d1 do base-old).")
    p.add_argument("--base-new", required=True, help="Repo ID ou caminho local do Base continuamente pré-treinado (geração d1d2).")

    # >>> NOVO: revisões (commit/tag/branch) opcionais <<<
    p.add_argument("--base-old-revision", default=None, help="Revision (commit/tag/branch) para carregar o base-old.")
    p.add_argument("--instruct-old-revision", default=None, help="Revision (commit/tag/branch) para carregar o instruct-old.")
    p.add_argument("--base-new-revision", default=None, help="Revision (commit/tag/branch) para carregar o base-new.")

    p.add_argument("--out-repo", required=True, help="Repo ID de saída no Hub, ex.: user/nome-modelo.")
    p.add_argument("--out-dir", default=None, help="Diretório local para salvar antes do push (default: ./<nome do repo>).")
    p.add_argument("--alpha", type=float, default=1.0, help="Escala do delta (default: 1.0).")
    p.add_argument("--private", action="store_true", help="Cria/push como privado.")
    p.add_argument("--trust-remote-code", action="store_true", help="Passa trust_remote_code=True ao carregar.")
    p.add_argument("--no-login", action="store_true", help="Não chamar login(); use HF_TOKEN já configurado.")
    p.add_argument("--dtype-base-new", default="bfloat16", choices=["float32","bfloat16","float16"],
                   help="Dtype do modelo base-new ao carregar (apenas para memória/armazenamento; cálculos de delta são em float32).")
    return p.parse_args()

def str_to_dtype(s: str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[s]

def _effective_revision(repo_or_path: str, revision: Optional[str]) -> Optional[str]:
    """
    Se for caminho local, ignora revision (não faz sentido no from_pretrained).
    """
    if revision is None:
        return None
    try:
        if Path(repo_or_path).exists():
            print(f">> Aviso: '{repo_or_path}' parece ser caminho local; ignorando revision='{revision}'.", flush=True)
            return None
    except Exception:
        # se der algum problema ao checar, só tenta usar revision mesmo
        return revision
    return revision

def load_state_dict(repo_or_path: str, trust_remote_code: bool, revision: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Carrega um modelo só para obter seu state_dict (em CPU/float32) e liberar o objeto depois.
    """
    rev = _effective_revision(repo_or_path, revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo_or_path,
        trust_remote_code=trust_remote_code,
        revision=rev,                      # <<< NOVO
        torch_dtype=torch.float32,         # para delta estável
        low_cpu_mem_usage=True,
        device_map=None
    )
    sd = {k: v.to("cpu", dtype=torch.float32) for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd

def main():
    args = parse_args()
    if not args.no_login:
        login(token=args.hf_token)

    # --- 1) Carrega state_dicts do par antigo (base_old e instruct_old) em float32/CPU ---
    print(">> Carregando base_old (CPU/float32)...", flush=True)
    sd_base_old = load_state_dict(args.base_old, args.trust_remote_code, revision=args.base_old_revision)

    print(">> Carregando instruct_old (CPU/float32)...", flush=True)
    sd_inst_old = load_state_dict(args.instruct_old, args.trust_remote_code, revision=args.instruct_old_revision)

    # Checagem básica de compatibilidade
    inter_keys = set(sd_base_old.keys()) & set(sd_inst_old.keys())
    if not inter_keys:
        print("ERRO: Nenhuma chave em comum entre base_old e instruct_old. Verifique se são da MESMA família/arquitetura.", file=sys.stderr)
        sys.exit(1)

    # --- 2) Carrega o base_new (em dtype escolhido para reduzir RAM no save) ---
    print(">> Carregando base_new...", flush=True)
    dtype_target = str_to_dtype(args.dtype_base_new)
    rev_base_new = _effective_revision(args.base_new, args.base_new_revision)
    model_new = AutoModelForCausalLM.from_pretrained(
        args.base_new,
        trust_remote_code=args.trust_remote_code,
        revision=rev_base_new,             
        torch_dtype=dtype_target,          
        low_cpu_mem_usage=True,
        device_map=None
    )
    sd_new = model_new.state_dict()  # Tensors possivelmente em bf16/fp16

    # --- 3) Aplica o delta no state_dict do base_new ---
    print(">> Aplicando delta (instruct_old - base_old) com alpha={:.4f}...".format(args.alpha), flush=True)
    updated_sd: Dict[str, torch.Tensor] = {}
    n_applied, n_skipped_shape, n_missing = 0, 0, 0

    for k, w_new in tqdm(sd_new.items(), desc="Somando delta", total=len(sd_new)):
        if k in inter_keys:
            w_bold = sd_base_old[k]
            w_iold = sd_inst_old[k]
            if w_bold.shape == w_iold.shape == w_new.shape:
                delta = (w_iold - w_bold)  # float32
                merged = (w_new.float() + args.alpha * delta).to(w_new.dtype)
                updated_sd[k] = merged
                n_applied += 1
            else:
                updated_sd[k] = w_new
                n_skipped_shape += 1
        else:
            updated_sd[k] = w_new
            n_missing += 1

    print(f">> Feito. Aplicadas: {n_applied}  |  Shapes incompatíveis: {n_skipped_shape}  |  Chaves sem delta: {n_missing}")

    missing, unexpected = model_new.load_state_dict(updated_sd, strict=False)
    if missing or unexpected:
        print(f">> Aviso load_state_dict: missing={len(missing)} | unexpected={len(unexpected)}", flush=True)

    # Limpa SDs antigos para liberar memória
    del sd_base_old, sd_inst_old, sd_new, updated_sd
    gc.collect()

    # --- 3.1) Herdar CONFIG do instruct_old (inclui chat_template etc.) ---
    try:
        print(">> Herdando config (incluindo chat_template) do instruct_old...", flush=True)
        rev_inst_old = _effective_revision(args.instruct_old, args.instruct_old_revision)
        inst_config = AutoConfig.from_pretrained(
            args.instruct_old,
            trust_remote_code=args.trust_remote_code,
            revision=rev_inst_old,          # <<< NOVO
        )
        model_new.config = inst_config
    except Exception as e:
        print(f">> Aviso: não foi possível carregar config do instruct_old ({e})")

    # --- 4) Salva localmente e faz push para o Hub ---
    out_dir = args.out_dir or Path(".") / args.out_repo.split("/")[-1]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Salvando em: {out_dir.resolve()}", flush=True)
    model_new.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")

    # Salva tokenizer e generation_config do INSTRUCT_OLD (na mesma revision, se fornecida)
    print(">> Salvando tokenizer e generation_config do instruct_old...", flush=True)
    tok = None
    gen = None
    try:
        rev_inst_old = _effective_revision(args.instruct_old, args.instruct_old_revision)
        tok = AutoTokenizer.from_pretrained(
            args.instruct_old,
            trust_remote_code=args.trust_remote_code,
            revision=rev_inst_old,          # <<< NOVO
        )
        tok.save_pretrained(out_dir)
    except Exception as e:
        print(f">> Aviso: falha ao salvar tokenizer ({e})")

    try:
        rev_inst_old = _effective_revision(args.instruct_old, args.instruct_old_revision)
        gen = GenerationConfig.from_pretrained(
            args.instruct_old,
            revision=rev_inst_old,          # <<< NOVO
        )
        gen.save_pretrained(out_dir)
    except Exception:
        gen = None

    # Push para o Hub
    print(f">> Fazendo push para o Hub em {args.out_repo} ...", flush=True)
    model_new.push_to_hub(
        args.out_repo,
        private=args.private,
        commit_message="Instruction Residual merge (base_new + alpha*(instruct_old-base_old))"
    )

    try:
        if tok is not None:
            tok.push_to_hub(
                args.out_repo,
                private=args.private,
                commit_message="Add tokenizer (from instruct_old)"
            )
    except Exception:
        pass

    try:
        if gen is not None:
            gen.push_to_hub(
                args.out_repo,
                private=args.private,
                commit_message="Add generation config (from instruct_old)"
            )
    except Exception:
        pass

    print(">> Concluído com sucesso.")

if __name__ == "__main__":
    main()
