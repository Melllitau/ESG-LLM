#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from statistics import mean
from collections import defaultdict
import numpy as np 
from collections import Counter 

import evaluate
from scipy.stats import ttest_rel

# =========================
# Config
# =========================
# Raiz onde o run_all.sh salvou tudo
RESULTS_ROOT = Path("./results_hf").resolve()

# Opcional: limite a avaliação a um subconjunto de modelos (nomes Ollama)
# Se vazio, o script descobre automaticamente pelos diretórios em RESULTS_ROOT
MODELS: List[str] = [
    # "gemma3:12b", "qwen3:8b", ...
]

# Modelos/pastas para pular durante a avaliação (nomes sanitizados ou originais)
SKIP_MODELS: List[str] = [
 "_eval_summary",
]

# Avaliar estes modos (ajuste se quiser só um)
MODES = ["llm", "agent"]

# Onde salvar o sumário global
SUMMARY_DIR = RESULTS_ROOT / "_eval_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_TXT = SUMMARY_DIR / "summary_results.txt"
SUMMARY_JSON = SUMMARY_DIR / "summary_results.json"

# =========================
# Helpers
# =========================
def to_json_safe(obj):
    """Recursively convert numpy types, defaultdicts, sets, etc. to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, set):
        return [to_json_safe(v) for v in obj]  # or sorted(...) if order matters
    return obj


def sanitize_model_name(tag: str) -> str:
    # mesmo sanitizador do run_all.sh: troca ':', '/', '.' por '_'
    return re.sub(r"[/:.]", "_", tag)

def discover_models(root: Path) -> List[str]:
    """Descobre modelos olhando os subdirs imediatos de results/, dessanitizando não é necessário.
    Usaremos a pasta já sanitizada como "id" e manteremos 'tag' desconhecido; tudo funciona com o sanitizado."""
    out = []
    skip_sanitized = [sanitize_model_name(s) for s in SKIP_MODELS]
    
    for p in sorted(root.iterdir()):
        if p.name in SKIP_MODELS or p.name in skip_sanitized:
            print(f"[SKIP] Pulando pasta: {p.name}")
            continue
        if p.is_dir() and (p / "llm").exists() or (p / "agent").exists():
            out.append(p.name)  # já vem sanitizado
    return out

def path_for(model_tag_or_sanitized: str, mode: str) -> Tuple[Path, Path]:
    """
    Retorna (dir_do_modo, caminho_json_resultado) seguindo o layout:
      results/<sanitized>/<mode>/<sanitized>_<mode>.json
    Aceita tanto um tag (gemma3:12b) quanto o nome já sanitizado.
    """
    name = sanitize_model_name(model_tag_or_sanitized)
    mode_dir = RESULTS_ROOT / name / mode
    json_path = mode_dir / f"{name}_{mode}.json"
    return mode_dir, json_path

# =========================
# Métricas
# =========================
SAMPLE_METRICS = [
    "BERTScore_precision",
    "BERTScore_recall",
    "BERTScore_f1",
    "METEOR",
    "BLEU",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "ROUGE-Lsum",
    "OverlapPrecision",
    "OverlapRecall",
    "OverlapF1",
]

bertscore_metric = evaluate.load("bertscore")
meteor_metric    = evaluate.load("meteor")
bleu_metric      = evaluate.load("bleu")
rouge_metric     = evaluate.load("rouge")

def evaluate_file(json_path: Path) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Lê o JSON (lista de amostras do run.py) e computa:
      - BERTScore, METEOR, BLEU, ROUGE, Overlap P/R/F1 por amostra
      - agregados (média)
    Espera chaves do run.py: 'generated_answer' e 'reference_answer'
    """
    raw_data = json.loads(json_path.read_text(encoding="utf-8"))

    # ==========================================================
    # MODIFICAÇÃO PARA TRATAR OS DOIS FORMATOS DE JSON
    # ==========================================================
    if isinstance(raw_data, dict) and 'results' in raw_data:
        # Formato 1: O JSON é um dicionário com a chave 'results'
        data = raw_data['results']
    elif isinstance(raw_data, list):
        # Formato 2: O JSON já é a lista de resultados (formato original)
        data = raw_data
    else:
        # Se não for nenhum dos formatos esperados
        raise ValueError(f"Formato de JSON inesperado no arquivo: {json_path}. Esperado lista ou dict com chave 'results'.")
    # ==========================================================

    # Mapeia campos do run.py
    all_generated = [d.get("generated_answer", "") or "" for d in data]
    all_refs      = [d.get("reference_answer", "") or "" for d in data]

    # ... o restante da função continua inalterado ...
    
    bert = bertscore_metric.compute(predictions=all_generated, references=all_refs, lang="en")

    meteor_scores, bleu_scores, rouge_scores = [], [], []
    overlap_precisions, overlap_recalls, overlap_f1s = [], [], []

    for gen_text, ref_text in zip(all_generated, all_refs):
        # Se a sentença gerada for vazia, atribui scores 0 para evitar erros
        if not gen_text.strip():
            meteor_scores.append(0.0)
            bleu_scores.append(0.0)
            rouge_scores.append({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0})
            overlap_precisions.append(0.0)
            overlap_recalls.append(0.0)
            overlap_f1s.append(0.0)
            continue

        # METEOR
        meteor_res = meteor_metric.compute(predictions=[gen_text], references=[ref_text])
        meteor_scores.append(float(meteor_res["meteor"]))

        # BLEU (note referências são [[ref]])
        bleu_res = bleu_metric.compute(predictions=[gen_text], references=[[ref_text]])
        bleu_scores.append(float(bleu_res["bleu"]))

        # ROUGE
        rouge_res = rouge_metric.compute(predictions=[gen_text], references=[ref_text])
        # rouge retorna valores já agregados por par
        rouge_scores.append({
            "rouge1":  float(rouge_res["rouge1"]),
            "rouge2":  float(rouge_res["rouge2"]),
            "rougeL":  float(rouge_res["rougeL"]),
            "rougeLsum": float(rouge_res["rougeLsum"]),
        })

        # Overlap token a token (bem simples)
        gen_tokens = gen_text.split()
        ref_tokens = ref_text.split()
        
        common = Counter(gen_tokens) & Counter(ref_tokens)  
        tp = sum(common.values())
        
        precision = tp / len(gen_tokens) if gen_tokens else 0.0
        recall    = tp / len(ref_tokens) if ref_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        overlap_precisions.append(precision)
        overlap_recalls.append(recall)
        overlap_f1s.append(f1)

    per_sample = []
    for i, entry in enumerate(data):
        sample_metrics = {
            "BERTScore_precision": float(bert["precision"][i]),
            "BERTScore_recall":    float(bert["recall"][i]),
            "BERTScore_f1":        float(bert["f1"][i]),
            "METEOR":              float(meteor_scores[i]),
            "BLEU":                float(bleu_scores[i]),
            "ROUGE-1":             float(rouge_scores[i]["rouge1"]),
            "ROUGE-2":             float(rouge_scores[i]["rouge2"]),
            "ROUGE-L":             float(rouge_scores[i]["rougeL"]),
            "ROUGE-Lsum":          float(rouge_scores[i]["rougeLsum"]),
            "OverlapPrecision":    float(overlap_precisions[i]),
            "OverlapRecall":       float(overlap_recalls[i]),
            "OverlapF1":           float(overlap_f1s[i]),
        }
        out_entry = dict(entry)
        out_entry["metrics"] = sample_metrics
        per_sample.append(out_entry)

    N = len(per_sample)
    overall = {}
    if N > 0:
        overall["BERTScore_precision"] = float(sum(bert["precision"]) / N)
        overall["BERTScore_recall"]    = float(sum(bert["recall"]) / N)
        overall["BERTScore_f1"]        = float(sum(bert["f1"]) / N)
        overall["METEOR"]              = float(sum(meteor_scores) / N)
        overall["BLEU"]                = float(sum(bleu_scores) / N)
        overall["ROUGE-1"]             = float(sum(r["rouge1"] for r in rouge_scores) / N)
        overall["ROUGE-2"]             = float(sum(r["rouge2"] for r in rouge_scores) / N)
        overall["ROUGE-L"]             = float(sum(r["rougeL"] for r in rouge_scores) / N)
        overall["ROUGE-Lsum"]          = float(sum(r["rougeLsum"] for r in rouge_scores) / N)
        overall["OverlapPrecision"]    = float(sum(overlap_precisions) / N)
        overall["OverlapRecall"]       = float(sum(overlap_recalls) / N)
        overall["OverlapF1"]           = float(sum(overlap_f1s) / N)
        overall["AvgGeneratedLength"]  = float(sum(len(t.split()) for t in all_generated) / N)

    return overall, per_sample

# =========================
# Loop principal
# =========================
# Descobrir modelos, se não especificados
if not MODELS:
    MODELS = discover_models(RESULTS_ROOT)
else:
    # Se MODELS foi especificado manualmente, ainda aplica o filtro de SKIP_MODELS
    skip_sanitized = [sanitize_model_name(s) for s in SKIP_MODELS]
    MODELS = [m for m in MODELS if m not in SKIP_MODELS and sanitize_model_name(m) not in skip_sanitized]

# Armazena per-sample para t-tests comparáveis por modo
# dict[mode][model] = per_sample_list
per_sample_store: Dict[str, Dict[str, List[Dict[str, Any]]]] = {m: {} for m in MODES}
overall_store: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in MODES}

for model in MODELS:
    name = sanitize_model_name(model)  # se já veio sanitizado, não muda
    for mode in MODES:
        mode_dir, json_path = path_for(name, mode)
        if not json_path.exists():
            print(f"[WARN] Missing file: {json_path}")
            continue

        print(f"[EVAL] {name} / {mode} -> {json_path}")
        overall, per_sample = evaluate_file(json_path)

        # Salva avaliação ao lado do JSON do modo
        eval_out = mode_dir / f"{name}_{mode}_evaluate_ALLMETRICS.json"
        with eval_out.open("w", encoding="utf-8") as f:
            json.dump({"overall_metrics": overall, "entries": per_sample}, f, ensure_ascii=False, indent=2)
        print(f"       saved: {eval_out}")

        per_sample_store[mode][name] = per_sample
        overall_store[mode][name] = overall

# =========================
# Sumário + significância por modo
# =========================
with SUMMARY_TXT.open("w", encoding="utf-8") as ftxt:
    summary_payload = {"modes": {}, "notes": "Paired t-test only across models that share identical sample counts."}

    for mode in MODES:
        models_here = sorted(overall_store[mode].keys())
        if not models_here:
            continue

        ftxt.write(f"# MODE: {mode}\n\n")
        summary_payload["modes"][mode] = {"overall_avgs": {}, "ttests": {}}

        # Tabela de médias por modelo
        for m in models_here:
            ftxt.write(f"## Model: {m}\n")
            summary_payload["modes"][mode]["overall_avgs"][m] = overall_store[mode][m]
            for met in SAMPLE_METRICS:
                val = overall_store[mode][m].get(met, None)
                if val is not None:
                    ftxt.write(f"  {met}: {val:.4f}\n")
            ftxt.write("\n")

        # T-tests pareados (por amostra) entre modelos do mesmo modo
        ftxt.write(f"# Statistical Significance (Paired t-test) — mode={mode}\n\n")
        ttest_results = defaultdict(dict)

        for i in range(len(models_here)):
            for j in range(i + 1, len(models_here)):
                m1, m2 = models_here[i], models_here[j]
                # Só testa se o número de amostras for igual
                s1 = per_sample_store[mode].get(m1, [])
                s2 = per_sample_store[mode].get(m2, [])
                if len(s1) == 0 or len(s1) != len(s2):
                    ftxt.write(f"{m1} vs {m2}: Not comparable (len {len(s1)} vs {len(s2)})\n")
                    continue

                ftxt.write(f"## {m1} vs {m2}\n")
                ttest_results[m1][m2] = {}
                for metric in SAMPLE_METRICS:
                    v1 = [row["metrics"][metric] for row in s1]
                    v2 = [row["metrics"][metric] for row in s2]
                    try:
                        tstat, pval = ttest_rel(v1, v2)
                        tstat = float(pval) if np.isnan(pval) else float(tstat)
                        pval = float(pval)  
                        signif = pval < 0.05  
                        ftxt.write(f"  {metric}: p={pval:.6f} => {'SIGNIFICANT' if signif else 'NOT-SIGNIFICANT'}\n")
                        ttest_results[m1][m2][metric] = {
                            "p": pval,
                            "t": tstat,
                            "significant_0.05": signif
                        }
                    except Exception as e:
                        ftxt.write(f"  {metric}: ERROR ({e})\n")
                ftxt.write("\n")

        summary_payload["modes"][mode]["ttests"] = ttest_results
        ftxt.write("\n\n")

    ftxt.write("DONE.\n")

with SUMMARY_JSON.open("w", encoding="utf-8") as fj:
    json.dump(to_json_safe(summary_payload), fj, ensure_ascii=False, indent=2)

print(f"\nSummary written to:\n - {SUMMARY_TXT}\n - {SUMMARY_JSON}\n")
