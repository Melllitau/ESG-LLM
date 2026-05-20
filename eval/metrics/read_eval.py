import os
import json
from pathlib import Path
from typing import List
import textstat

# =========================
# Config e helpers do eval.py
# =========================
RESULTS_ROOT = Path("./results").resolve()

def sanitize_model_name(tag: str) -> str:
    import re
    return re.sub(r"[/:.]", "_", tag)

def discover_models(root: Path) -> List[str]:
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and ((p / "llm").exists() or (p / "agent").exists() or (p / "search-agent-1").exists()):
            out.append(p.name)
    return out

def path_for(model_tag_or_sanitized: str, mode: str):
    name = sanitize_model_name(model_tag_or_sanitized)
    mode_dir = RESULTS_ROOT / name / mode
    # Special case: search-agent-1 folder contains files named with "_agent_"
    file_mode = "agent" if mode == "search-agent-1" else mode
    json_path = mode_dir / f"{name}_{file_mode}_evaluate_ALLMETRICS.json"
    return mode_dir, json_path

# =========================
# Modos e modelos
# =========================
MODES = ["llm", "agent", "search-agent-1"]
MODELS = discover_models(RESULTS_ROOT)

for model in MODELS:
    name = sanitize_model_name(model)
    for mode in MODES:
        mode_dir, json_path = path_for(name, mode)
        if not json_path.exists():
            print(f"[WARN] Missing file: {json_path}")
            continue

        # Listas para calcular médias de TODAS as métricas
        gen_metrics_lists = {}
        ref_metrics_lists = {}
        
        print(f"[TEXTSTAT] {name} / {mode} -> {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        for entry in entries:
            gen = entry.get("generated_answer", "")
            ref = entry.get("reference_answer", "")
            metrics = entry.get("metrics", {})

            # Métricas textstat para resposta gerada
            gen_metrics = {
                "gen_syllable_count": textstat.syllable_count(gen),
                "gen_sentence_count": textstat.sentence_count(gen),
                "gen_flesch_reading_ease": textstat.flesch_reading_ease(gen),
                "gen_flesch_kincaid_grade": textstat.flesch_kincaid_grade(gen),
                "gen_smog_index": textstat.smog_index(gen),
                "gen_coleman_liau_index": textstat.coleman_liau_index(gen),
                "gen_automated_readability_index": textstat.automated_readability_index(gen),
                "gen_dale_chall_readability_score": textstat.dale_chall_readability_score(gen),
                "gen_difficult_words": textstat.difficult_words(gen),
                "gen_linsear_write_formula": textstat.linsear_write_formula(gen),
                "gen_gunning_fog": textstat.gunning_fog(gen),
                "gen_text_standard": textstat.text_standard(gen),
                "gen_fernandez_huerta": textstat.fernandez_huerta(gen),
                "gen_szigriszt_pazos": textstat.szigriszt_pazos(gen),
                "gen_gutierrez_polini": textstat.gutierrez_polini(gen),
                "gen_crawford": textstat.crawford(gen),
                "gen_gulpease_index": textstat.gulpease_index(gen),
                "gen_osman": textstat.osman(gen)
            }
            
            # Adiciona ao metrics e às listas para média
            for k, v in gen_metrics.items():
                metrics[k] = v
                if k not in gen_metrics_lists:
                    gen_metrics_lists[k] = []
                gen_metrics_lists[k].append(v)

            # Métricas textstat para resposta de referência
            ref_metrics = {
                "ref_syllable_count": textstat.syllable_count(ref),
                "ref_sentence_count": textstat.sentence_count(ref),
                "ref_flesch_reading_ease": textstat.flesch_reading_ease(ref),
                "ref_flesch_kincaid_grade": textstat.flesch_kincaid_grade(ref),
                "ref_smog_index": textstat.smog_index(ref),
                "ref_coleman_liau_index": textstat.coleman_liau_index(ref),
                "ref_automated_readability_index": textstat.automated_readability_index(ref),
                "ref_dale_chall_readability_score": textstat.dale_chall_readability_score(ref),
                "ref_difficult_words": textstat.difficult_words(ref),
                "ref_linsear_write_formula": textstat.linsear_write_formula(ref),
                "ref_gunning_fog": textstat.gunning_fog(ref),
                "ref_text_standard": textstat.text_standard(ref),
                "ref_fernandez_huerta": textstat.fernandez_huerta(ref),
                "ref_szigriszt_pazos": textstat.szigriszt_pazos(ref),
                "ref_gutierrez_polini": textstat.gutierrez_polini(ref),
                "ref_crawford": textstat.crawford(ref),
                "ref_gulpease_index": textstat.gulpease_index(ref),
                "ref_osman": textstat.osman(ref)
            }
            
            # Adiciona ao metrics e às listas para média
            for k, v in ref_metrics.items():
                metrics[k] = v
                if k not in ref_metrics_lists:
                    ref_metrics_lists[k] = []
                ref_metrics_lists[k].append(v)

            entry["metrics"] = metrics
            
        # Cálculo das médias (overall)
        overall_textstat_metrics = {}
        if entries:
            # Calcula médias para métricas gen
            for metric_name, values_list in gen_metrics_lists.items():
                if values_list and all(isinstance(x, (int, float)) for x in values_list):
                    overall_textstat_metrics[metric_name] = sum(values_list) / len(values_list)
            
            # Calcula médias para métricas ref
            for metric_name, values_list in ref_metrics_lists.items():
                if values_list and all(isinstance(x, (int, float)) for x in values_list):
                    overall_textstat_metrics[metric_name] = sum(values_list) / len(values_list)

        # Adiciona as médias do textstat ao overall_metrics
        if "overall_metrics" not in data:
            data["overall_metrics"] = {}
        data["overall_metrics"].update(overall_textstat_metrics)

        # Salva em novo arquivo
        file_mode = "agent" if mode == "search-agent-1" else mode
        new_path = mode_dir / f"{name}_{file_mode}_evaluate_ALLMETRICS_TEXTSTAT.json"
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"       saved: {new_path}")

print("Concluído.")