'''
python run_hf.py \
  --mode agent \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/esg_qa_dev.jsonl \
  --use_ir \
  --embed_model BAAI/bge-small-en-v1.5 \
  --persist_dir ./ir_index_esg \
  --top_k 5 \
  --output qwen25_7b_agent_ir.json
  
python run_hf.py \
  --mode llm \
  --model g-assismoraes/Qwen3-4B-irm-esg \
  --input data.json \
  --output Qwen3-4B-irm-esg_llm_only.json


'''

import asyncio
import argparse
import json
import os
import re
import time
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.agent.workflow import ReActAgent

import eco2ai

from huggingface_hub import login
#HUGGING FACE TOKEN HERE
login(token="")

MY_CHAT_TEMPLATE = '''{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}'''


# ---------------------------
# Utilities: data loading & docs
# ---------------------------
def save_results_with_tracker_info(results: List[Dict[str, Any]], output_path: str, tracker=None):
    """Salva os resultados em arquivo JSON com informações do tracker para retomada."""
    try:
        # Remove duplicatas mais precisas baseado em pergunta + resposta gerada
        unique_results = []
        seen_pairs = set()
        
        for result in results:
            question = result.get("question", "").strip()
            generated = result.get("generated_answer", "").strip()
            pair_key = (question, generated[:100])  # Usa apenas os primeiros 100 chars da resposta
            
            if question and pair_key not in seen_pairs:
                unique_results.append(result)
                seen_pairs.add(pair_key)
        
        if len(unique_results) != len(results):
            print(f"[SAVE] Removidas {len(results) - len(unique_results)} entradas duplicadas antes do salvamento")
        
        # Estrutura do arquivo de saída com metadados do tracker
        output_data = {
            "results": unique_results,
            "metadata": {
                "total_processed": len(unique_results),
                "last_update": time.time(),
                "tracker_info": {}
            }
        }
        
        # Captura informações do tracker se disponível
        if tracker is not None:
            try:
                current_cumulative_energy = getattr(tracker, '_cumulative_energy', 0)
                current_cumulative_co2 = getattr(tracker, '_cumulative_co2', 0)

                tracker_data = {
                    "project_name": getattr(tracker, 'project_name', ''),
                    "experiment_description": getattr(tracker, 'experiment_description', ''),
                    "start_time": getattr(tracker, '_start_time', None),
                    "energy_consumption": current_cumulative_energy,
                    "co2_emission": current_cumulative_co2,
                }
                
                if current_cumulative_energy > 0 or current_cumulative_co2 > 0:
                    print(f"[SAVE] Métricas eco cumulativas salvas: energia={current_cumulative_energy:.4f}, CO2={current_cumulative_co2:.4f}")

                output_data["metadata"]["tracker_info"] = tracker_data
            except Exception as e:
                print(f"[SAVE] Aviso: Não foi possível capturar informações do tracker: {e}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Resultados salvos: {len(unique_results)} itens únicos em {output_path}")
    except Exception as e:
        print(f"[SAVE] Erro ao salvar resultados: {e}")

def save_results_periodically(results: List[Dict[str, Any]], output_path: str, tracker=None):
    """Salva os resultados em arquivo JSON."""
    if tracker is not None:
        save_results_with_tracker_info(results, output_path, tracker)
    else:
        # Fallback para versão simples se não há tracker
        try:
            unique_results = []
            seen_pairs = set()
            
            for result in results:
                question = result.get("question", "").strip()
                generated = result.get("generated_answer", "").strip()
                pair_key = (question, generated[:100])
                
                if question and pair_key not in seen_pairs:
                    unique_results.append(result)
                    seen_pairs.add(pair_key)
            
            if len(unique_results) != len(results):
                print(f"[SAVE] Removidas {len(results) - len(unique_results)} entradas duplicadas antes do salvamento")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(unique_results, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] Resultados salvos: {len(unique_results)} itens únicos em {output_path}")
        except Exception as e:
            print(f"[SAVE] Erro ao salvar resultados: {e}")

def load_existing_results(output_path: str) -> Tuple[List[Dict[str, Any]], set, Dict[str, Any]]:
    """Carrega resultados existentes e metadados do tracker de um arquivo JSON, se existir.
    Retorna (resultados, set_de_perguntas_processadas, tracker_metadata) para eficiência."""
    try:
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Verifica se é o novo formato com metadata ou formato antigo
            if isinstance(data, dict) and "results" in data:
                # Novo formato
                results = data["results"]
                metadata = data.get("metadata", {})
                tracker_info = metadata.get("tracker_info", {})
            else:
                # Formato antigo - compatibilidade
                results = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
                tracker_info = {}
                
            # Cria o set de perguntas processadas imediatamente
            processed_questions = {
                result.get("question", "").strip() 
                for result in results 
                if result.get("question", "").strip()
            }
            print(f"[RESUME] Carregados {len(results)} resultados existentes ({len(processed_questions)} perguntas únicas) de {output_path}")
            if tracker_info:
                print(f"[RESUME] Informações do tracker recuperadas para continuação")
            return results, processed_questions, tracker_info
        return [], set(), {}
    except Exception as e:
        print(f"[RESUME] Erro ao carregar resultados existentes: {e}")
        return [], set(), {}

def load_items(input_path: str) -> List[Dict[str, Any]]:
    """Load items from a JSON or JSONL file. Supports one object or a list of objects."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = p.read_text(encoding="utf-8").strip()
    # Try JSON array or single JSON object
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try JSONL
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def make_hf_llm(model_name: str) -> HuggingFaceLLM:
    """
    Helper to build a HuggingFaceLLM. Adjust model_kwargs/generate_kwargs as needed.
    """
    llm = HuggingFaceLLM(
        model_name=model_name,
        # You can add tokenizer_name=model_name explicitly if needed:
        tokenizer_name=model_name,
        # Basic defaults – tweak if you want:
        context_window=4096,
        max_new_tokens=10000,
        generate_kwargs={"temperature": 0.8},
        device_map="auto",
        model_kwargs={"load_in_4bit": True,"trust_remote_code": True},  # uncomment if your model needs this
    )
    return llm


def configure_llamaindex(model_name: str, embed_model_name: str):
    # Define a default HF LLM + HF embeddings (no OpenAI, no Ollama)
    Settings.llm = make_hf_llm(model_name)
    #Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)


def build_documents(items: List[Dict[str, Any]]) -> List[Document]:
    """Build LlamaIndex Documents from the contexts in items."""
    docs: List[Document] = []
    for i, ex in enumerate(items):
        ctx = ex.get("context", "")
        meta = {
            "qid": i,
            "source_file": ex.get("source_file", ""),
            "stringFilter": ex.get("stringFilter", ""),
            "isAlign": ex.get("isAlign", None),
            "question": ex.get("question", ""),
        }
        if ctx and ctx.strip():
            docs.append(Document(text=ctx, metadata=meta, doc_id=f"doc-{i}"))
    return docs


# ---------------------------
# IR index/tool (create-if-not-exist) 
# ---------------------------
def build_or_load_ir(
    items: List[Dict[str, Any]],
    embed_model_name: str,
    persist_dir: str,
    top_k: int,
    tool_name: str = "search",
    dedupe: bool = True,
    dedupe_case_insensitive: bool = True,
):
    """
    Cria (ou carrega) um índice vetorial sobre os 'context' do JSON 
    e expõe um FunctionTool que retorna os top-k trechos em texto puro.
    """
    import hashlib

    #Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    pdir = Path(persist_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    index = None
    # Tenta carregar se houver algo na pasta; se falhar, reconstrói
    try:
        has_any = any(pdir.iterdir())
    except Exception:
        has_any = False

    if has_any:
        try:
            sc = StorageContext.from_defaults(persist_dir=str(pdir))
            index = load_index_from_storage(sc)
        except Exception as e:
            print(f"[IR] Falha ao carregar índice existente ({e}). Reconstruindo...")
            index = None

    if index is None:
        # --- Construção do índice a partir de contextos ÚNICOS ---
        def _norm_ctx(s: str) -> str:
            if s is None:
                return ""
            # trim + colapso de espaços
            base = " ".join(s.split())
            return base.lower() if dedupe_case_insensitive else base

        unique_map: Dict[str, Dict[str, Any]] = {}  # key_norm -> {"text": str, "meta": dict}
        for i, ex in enumerate(items):
            raw_ctx = (ex.get("context") or "").strip()
            if not raw_ctx:
                continue
            key = _norm_ctx(raw_ctx) if dedupe else raw_ctx

            if key not in unique_map:
                # metadados iniciais 
                meta = {
                    "qid": i,  # compat: qid "principal"
                    "qid_first": i,
                    "qid_list": [i],
                    "source_file": ex.get("source_file", ""),  # compat: source_file "principal"
                    "source_file_first": ex.get("source_file", ""),
                    "source_files": [ex.get("source_file", "")] if ex.get("source_file") else [],
                    "stringFilter_first": ex.get("stringFilter", ""),
                    "stringFilters": [ex.get("stringFilter", "")] if ex.get("stringFilter") else [],
                    "isAlign_list": [ex.get("isAlign", None)],
                    "question_first": ex.get("question", ""),
                    "dup_count": 1,
                }
                unique_map[key] = {"text": raw_ctx, "meta": meta}
            else:
                entry = unique_map[key]
                m = entry["meta"]
                m["qid_list"].append(i)
                sf = ex.get("source_file", "")
                if sf:
                    m["source_files"].append(sf)
                sfilt = ex.get("stringFilter", "")
                if sfilt:
                    m["stringFilters"].append(sfilt)
                m["isAlign_list"].append(ex.get("isAlign", None))
                m["dup_count"] += 1

        # Compacta listas (remove vazios/duplicados) e cria Documents
        docs: List[Document] = []
        for key_norm, entry in unique_map.items():
            meta = entry["meta"]
            # normaliza listas
            meta["source_files"] = sorted({x for x in meta.get("source_files", []) if x})
            meta["stringFilters"] = sorted({x for x in meta.get("stringFilters", []) if x})
            # doc_id estável baseado no conteúdo normalizado
            doc_id = f"doc-{hashlib.blake2b(key_norm.encode('utf-8'), digest_size=8).hexdigest()}"
            docs.append(Document(text=entry["text"], metadata=meta, doc_id=doc_id))

        if not docs:
            raise ValueError("Não há 'context' não-vazio para indexar.")

        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=str(pdir))

    # --- FunctionTool de retrieval puro ---
    retriever = index.as_retriever(similarity_top_k=top_k)

    def retrieve(query: str) -> str:
        """
        Dado um query, retorna os top-k trechos concatenados em texto,
        com metadados mínimos para rastreabilidade.
        """
        nodes = retriever.retrieve(query)
        blocks = []
        for rank, n in enumerate(nodes, start=1):
            meta = n.node.metadata or {}
            # Compat: usa 'source_file' e 'qid' preservados do primeiro item
            src = meta.get("source_file", "")
            qid = meta.get("qid", "")
            content = n.node.get_content()
            blocks.append(
                f"=== [#{rank}] score={getattr(n, 'score', 0.0):.4f} source={src} qid={qid} ===\n{content}"
            )
        return "\n\n".join(blocks) if blocks else "(sem resultados)"

    retriever_tool = FunctionTool.from_defaults(
        fn=retrieve,
        name=tool_name,
        description=f"Recupera top-{top_k} contextos ÚNICOS do dataset em texto plano; use antes de responder."
    )

    return retriever_tool, index

# ---------------------------
# Agent
# ---------------------------
def create_agent(model_name: str, tools: List[Any], force_system_prompt: bool = True) -> ReActAgent:
    """
    Create a ReAct agent. If tools contain a retriever tool, the agent can do RAG.
    """
    llm = make_hf_llm(model_name)

    sys_prompt = (
        "You are a question-answering assistant. "
        "If a search tool is available (e.g., ESGRetriever), always call it first "
        "to retrieve relevant context (at least top-3) and then compose your final answer from those contexts. "
        "If context is available, prioritize the information from the context over your prior knowledge. "
        "For any question with math, you must use external computation rather than doing math in your head."
    ) if force_system_prompt else None

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=sys_prompt,
    )
    return agent


# ---------------------------
# Evaluation Loop (AGENT)
# ---------------------------
async def evaluate_agent(
    agent: ReActAgent,
    items: List[Dict[str, Any]],
    use_ir: bool,
    index: VectorStoreIndex = None,
    top_k: int = 3,
    output_path: str = None,
    tracker=None,
) -> List[Dict[str, Any]]:
    # Carrega resultados existentes e set de perguntas processadas de uma vez
    results, processed_questions, _ = load_existing_results(output_path) if output_path else ([], set(), {})
    
    retriever = index.as_retriever(similarity_top_k=top_k) if (use_ir and index is not None) else None
    
    # Configuração para salvamento periódico (1 hora = 3600 segundos)
    save_interval = 3600  # 1 hora em segundos
    last_save_time = time.time()

    # Filtragem otimizada: usa lookup O(1) no set em vez de O(n) na lista
    remaining_items = []
    skipped_count = 0
    total_questions = 0
    
    for ex in items:
        question = (ex.get("question") or "").strip()
        if not question:
            continue
        total_questions += 1
        
        # Lookup O(1) no set em vez de busca linear na lista
        if question in processed_questions:
            skipped_count += 1
        else:
            remaining_items.append(ex)
    
    if skipped_count > 0:
        print(f"[RESUME] Pulando {skipped_count}/{total_questions} perguntas já processadas. Restam {len(remaining_items)} para processar.")
    
    if not remaining_items:
        print(f"[RESUME] Todas as {total_questions} perguntas já foram processadas! Total: {len(results)} resultados.")
        return results

    pbar = tqdm(remaining_items, total=len(remaining_items), desc="Executando (agent)", unit="q")
    for ex in pbar:
        question = ex.get("question", "").strip()
        reference_answer = ex.get("answer", "")
        ctx = ex.get("context", "")
        source_file = ex.get("source_file", "")

        if not question:
            continue

        # Retrieve contexts independentemente (para logging/avaliação)
        retrieved_contexts = []
        if retriever is not None:
            try:
                nodes = retriever.retrieve(question)
                for nw in nodes:
                    retrieved_contexts.append({
                        "text": nw.node.get_content(),
                        "score": nw.score,
                        "metadata": nw.node.metadata or {},
                    })
            except Exception as e:
                retrieved_contexts = [{"error": f"retrieval_failed: {e}"}]

        # Run the agent
        try:
            response = await agent.run(user_msg=question, max_iterations=20)
            print(response)
            raw_text = str(response)
            
            extra_think = ""
            generated = raw_text.strip()

            m = re.search(r"</think>", raw_text, flags=re.IGNORECASE)
            if m:
                # Tudo antes do </think> é o "thinking"
                extra_think = raw_text[:m.start()].strip()
                # Tudo depois do </think> é a resposta visível
                generated = raw_text[m.end():].strip()
            else:
                # fallback pro comportamento antigo, se você quiser manter
                think_blocks = re.findall(
                    r"<think>(.*?)</think>",
                    raw_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                extra_think = "\n\n".join(b.strip() for b in think_blocks) if think_blocks else ""
                generated = re.sub(
                    r"<think>.*?</think>",
                    "",
                    raw_text,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()

            tool_calls = str(getattr(response, "tool_calls", ""))
        except Exception as e:
            generated = f"Error during generation: {e}"
            extra_think = ""
            tool_calls = "[]"

        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated,   # sem <think>
            "extra_think": extra_think,      # apenas o conteúdo do <think>
            "context": ctx,
            "retrieved_contexts": retrieved_contexts,   # [] se use_ir=False
            "source_file": source_file,
            "tool_call": tool_calls,
        })

        # Verifica se é hora de salvar periodicamente
        current_time = time.time()
        if output_path and (current_time - last_save_time) >= save_interval:
            save_results_periodically(results, output_path, tracker)
            last_save_time = current_time

    return results


# ---------------------------
# Evaluation Loop (LLM-ONLY, sem agente/ferramentas)
# ---------------------------
async def evaluate_llm_only(
    llm: HuggingFaceLLM,
    items: List[Dict[str, Any]],
    output_path: str = None,
    tracker=None,
) -> List[Dict[str, Any]]:
    # Carrega resultados existentes e set de perguntas processadas de uma vez
    results, processed_questions, _ = load_existing_results(output_path) if output_path else ([], set(), {})
    
    # Configuração para salvamento periódico (1 hora = 3600 segundos)
    save_interval = 3600  # 1 hora em segundos
    last_save_time = time.time()

    # Filtragem otimizada: usa lookup O(1) no set em vez de O(n) na lista
    remaining_items = []
    skipped_count = 0
    total_questions = 0
    
    for ex in items:
        #CHANGE HERE QA
        question = (
            ex.get("question_pt_origin")
            or ex.get("question")
            or ""
        ).strip()
        if not question:
            continue
        total_questions += 1
        
        # Lookup O(1) no set em vez de busca linear na lista
        if question in processed_questions:
            skipped_count += 1
        else:
            remaining_items.append(ex)
    
    if skipped_count > 0:
        print(f"[RESUME] Pulando {skipped_count}/{total_questions} perguntas já processadas. Restam {len(remaining_items)} para processar.")
    
    if not remaining_items:
        print(f"[RESUME] Todas as {total_questions} perguntas já foram processadas! Total: {len(results)} resultados.")
        return results

    pbar = tqdm(remaining_items, total=len(remaining_items), desc="Executando (llm)", unit="q")
    for ex in pbar:
        #change HERE TO QA
        question = ex.get("question_pt_origin", "").strip()
        reference_answer = ex.get("answer_pt_origin", "")
        ctx = ex.get("abstract_translated_pt", "")
        #source_file = ex.get("source_file", "")

        if not question:
            continue

        try:
            messages = [
                ChatMessage(role="system", content="You are a concise question-answering assistant."),
                ChatMessage(role="user", content=question),
            ]
            # .chat é síncrono; usa thread para não travar o loop
            resp = await asyncio.to_thread(llm.chat, messages)
            raw_text = str(resp) #resp.message.content if hasattr(resp, "message") else str(resp)

            # separar <think>...</think>
            extra_think = ""
            generated = raw_text.strip()

            m = re.search(r"</think>", raw_text, flags=re.IGNORECASE)
            if m:
                # Tudo antes do </think> é o "thinking"
                extra_think = raw_text[:m.start()].strip()
                # Tudo depois do </think> é a resposta visível
                generated = raw_text[m.end():].strip()
            else:
                # fallback pro comportamento antigo, se você quiser manter
                think_blocks = re.findall(
                    r"<think>(.*?)</think>",
                    raw_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                extra_think = "\n\n".join(b.strip() for b in think_blocks) if think_blocks else ""
                generated = re.sub(
                    r"<think>.*?</think>",
                    "",
                    raw_text,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()

        except Exception as e:
            generated = f"Error during generation: {e}"
            extra_think = ""

        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated,
            "extra_think": extra_think,
            "context": ctx,
            "retrieved_contexts": [],   # sem IR
            #"source_file": source_file,
            "tool_call": "[]",          # sem ferramentas
        })

        # Verifica se é hora de salvar periodicamente
        current_time = time.time()
        if output_path and (current_time - last_save_time) >= save_interval:
            save_results_periodically(results, output_path, tracker)
            last_save_time = current_time

    return results


# ---------------------------
# Main
# ---------------------------
async def main(
    mode: str,
    model_name: str,
    input_path: str,
    output_path: str,
    use_ir: bool,
    embed_model_name: str,
    persist_dir: str,
    top_k: int,
    tracker,
):
    items = load_items(input_path)
    # Evita fallback para OpenAI em qualquer caminho
    configure_llamaindex(model_name, embed_model_name)

    if mode == "llm":
        # Sem agente, sem ferramentas
        llm = make_hf_llm(model_name)
        results = await evaluate_llm_only(llm=llm, items=items, output_path=output_path, tracker=tracker)

    else:
        # mode == "agent"
        tools: List[Any] = []
        index = None
        if use_ir:
            retr_tool, index = build_or_load_ir(
                items=items,
                embed_model_name=embed_model_name,
                persist_dir=persist_dir,
                top_k=top_k,
            )
            tools.append(retr_tool)

        agent = create_agent(model_name=model_name, tools=tools, force_system_prompt=True)
        results = await evaluate_agent(
            agent=agent,
            items=items,
            use_ir=use_ir,
            index=index,
            top_k=top_k,
            output_path=output_path,
            tracker=tracker,
        )

    # Salvamento final
    save_results_periodically(results, output_path, tracker)
    print(f"[FINAL] Processamento completo. Resultados finais salvos em {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agent", "llm"],
        default="agent",
        help="agent = ReAct Agent (com ou sem IR), llm = apenas LLM (sem ferramentas).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model id or local path (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct').",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON/JSONL with fields including 'question' and 'context'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (defaults to <model>_<mode>.json)",
    )
    parser.add_argument(
        "--use_ir",
        action="store_true",
        help="(Somente no modo 'agent') Habilita IR (RAG).",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Free HF embedding model for IR",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./ir_index",
        help="Directory to persist/load the IR index",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-k contexts to retrieve (IR)",
    )
    args = parser.parse_args()

    # Default output filename
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out = args.output or f"{safe_model}_{args.mode}.json"

    out_path = Path(out)
    eco_path = out_path.with_name(out_path.stem + "_eco.csv")

    tracker = eco2ai.Tracker(
        project_name="Itau Agents",
        experiment_description=f"QA | mode={args.mode} | model={args.model} | use_ir={args.use_ir}",
        file_name=str(eco_path),   # now saved next to the output JSON
    )
    tracker.start()

    try:
        asyncio.run(
            main(
                mode=args.mode,
                model_name=args.model,
                input_path=args.input,
                output_path=out,
                use_ir=args.use_ir,
                embed_model_name=args.embed_model,
                persist_dir=args.persist_dir,
                top_k=args.top_k,
                tracker=tracker,
            )
        )
    finally:
        tracker.stop()
