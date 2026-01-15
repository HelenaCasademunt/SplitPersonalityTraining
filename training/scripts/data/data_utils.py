import torch as t, json, os, requests, time
from scripts.utils import load_cfg
from transformers import AutoTokenizer


# ========================================================================================================================================================
# ========= QUERY OPENROUTER==============================================================================================================================
# ========================================================================================================================================================
def query_model(model, interaction_history, temperature: float = 0.001, max_tokens: int = 1024, reasoning : bool = True, return_cot: bool = False):
    api_key = os.environ.get('OPENROUTER_KEY')

    for attempt in range(50):
        payload = {
            "model": model,
            "messages": interaction_history,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {
                "enabled": reasoning,
                "exclude": False
            },
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload)
        )

        if response.status_code == 429:
            print(f"WARNING: OPEN ROUTER RATE LIMIT HIT {attempt+1}/50 — retrying…")
            time.sleep(attempt*2 + 2)
            continue
        
        if response.status_code in [400, 401, 408, 500, 501, 502, 503, 504]:
            print(f"WARNING: OPEN ROUTER SERVER ERROR {response.status_code} {attempt+1}/50 — retrying…")
            time.sleep(min(attempt*2 + 5, 60))  # Exponential backoff, max 60s
            continue

        response.raise_for_status()
        data = response.json()['choices'][0]['message']
        content = data.get('content')
        
        if (not content) or (isinstance(content, str) and content.strip() == ""):
            print(f"WARNING: EMPTY RESPONSE {attempt+1}/50 — retrying…")
            time.sleep(min(attempt*2 + 2, 30))
            continue

        if return_cot:
            return content, data.get('reasoning', None)

        time.sleep(0.2)
        return content
    raise RuntimeError(f"Failed to get response after 50 attempts")


def query_target(interaction_history, temperature=0.01, max_genlen=512):
    base_model = os.environ.get('base_model')
    return query_model(base_model, interaction_history, max_tokens = max_genlen, reasoning=False, return_cot=False, temperature=temperature)


def query_parser(question):
    parser_model = os.environ.get('parser_model')
    interaction_history = [{"role": "user", "content": f"{question}"}]
    return query_model(parser_model, interaction_history, reasoning=True, return_cot=False)


def query_reviewer(question):
    parser_model = os.environ.get('review_model')
    interaction_history = [{"role": "user", "content": f"{question}"}]
    return query_model(parser_model, interaction_history, reasoning=True, return_cot=False)




# ========================================================================================================================================================
# ==========  TOKENIZATION MADNESS  ======================================================================================================================
# ========================================================================================================================================================
def model_tokenization_type(model):
    d = {
        "meta-llama/Meta-Llama-3-8B-Instruct"        : (1,1,128002),
        "meta-llama/Meta-Llama-3-70B-Instruct"       : (1,1,128002),
        "meta-llama/Llama-3.1-8B-Instruct"           : (1,1,128002),    
        "meta-llama/Llama-3.1-70B-Instruct"          : (1,1,128002),   
        "meta-llama/Llama-3.1-405B-Instruct"         : (1,1,128002),
        "meta-llama/Llama-3.2-1B-Instruct"           : (1,1,128002),
        "meta-llama/Llama-3.2-3B-Instruct"           : (1,1,128002),
        "meta-llama/Llama-3.3-70B-Instruct"          : (1,1,128002),
        "google/gemma-3-4b-it"                       : (2,2,6),
        "google/gemma-3-12b-it"                      : (2,2,6),
        "google/gemma-3-27b-it"                      : (2,2,6),
        "Qwen/Qwen3-235B-A22B"                       : (2,2,128244),
        "Qwen/Qwen3-30B-A3B"                         : (2,2,128244),
        "Qwen/Qwen3-32B"                             : (2,2,128244),
        "Qwen/Qwen3-14B"                             : (2,2,128244),
        "Qwen/Qwen3-8B"                              : (2,2,128244),
        "Qwen/Qwen3-4B"                              : (2,2,128244),
        "Qwen/Qwen3-1.7B"                            : (2,2,128244),
        "Qwen/Qwen3-0.6B"                            : (2,2,128244),
        "Qwen/Qwen3-235B-A22B-Thinking-2507"         : (2,2,128244),
        "Qwen/Qwen3-235B-A22B-Instruct-2507"         : (2,2,128244),
        "Qwen/Qwen3-30B-A3B-Thinking-2507"           : (2,2,128244),
        "Qwen/Qwen3-30B-A3B-Instruct-2507"           : (2,2,128244),
        "Qwen/Qwen3-4B-Thinking-2507"                : (2,2,128244),
        "Qwen/Qwen3-4B-Instruct-2507"                : (2,2,128244),
        "Qwen/Qwen2.5-72B-Instruct"                  : (2,2,128244),
        "Qwen/Qwen2.5-32B-Instruct"                  : (2,2,128244),
        "openai/gpt-oss-20b"                         : (3,1,200017),
        "openai/gpt-oss-120b"                        : (3,1,200017),
        "databricks/dbrx-instruct"                   : (3,5,100275),
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"      : (4,3,128244),
        "deepseek-ai/DeepSeek-R1-0528"               : (4,3,128000),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  : (5,4,128244),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    : (5,4,128244),
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   : (5,4,128002),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"   : (5,4,128244),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   : (5,4,128244),
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  : (5,4,128002),
        "allenai/OLMo-2-0325-32B-Instruct"           : (6,6,100275),
        "allenai/OLMo-2-1124-13B-Instruct"           : (6,6,100275),
        "allenai/OLMo-2-1124-7B-Instruct"            : (6,6,100275),
        "01-ai/Yi-1.5-34B-Chat"                      : (7,7,13),
        "01-ai/Yi-1.5-9B-Chat"                       : (7,7,13),
        "zai-org/GLM-4.5"                            : (8,8,151301),
        "zai-org/GLM-4.5-Air"                        : (8,8,151301),
        "zai-org/GLM-4.1V-9B-Thinking"               : (8,8,151301),
        "zai-org/GLM-Z1-32B-0414"                    : (8,8,151301),
        "zai-org/GLM-Z1-9B-0414"                     : (8,8,151301),
        "zai-org/glm-4-9b-chat-hf"                   : (8,8,151301)
    }
    return d[model]


def get_hp_token_id(model):
    _, _, idx = model_tokenization_type(model)
    return idx


def get_user_prefix_tokens(tokenizer, model):
    tokens = tokenizer.apply_chat_template([
        {"role" : "user", "content" : "zero"},
        {"role" : "assistant", "content" : "zero"},
        {"role" : "user", "content" : "x"}])
    elicit_group, _, _ = model_tokenization_type(model)

    if elicit_group == 1:   return tokens[-6:-2]
    elif elicit_group == 2: return tokens[-6:-3]
    elif elicit_group == 3: return tokens[-5:-2]
    elif elicit_group == 4: return tokens[-3:-2]
    elif elicit_group == 5: return tokens[-2:-1]
    elif elicit_group == 6: return tokens[-7:-2]
    elif elicit_group == 7: return tokens[-9:-6]
    else:                   return tokens[-3:-1]

def get_assistant_prefix_tokens(tokenizer,model):
    _, eot_group, _ = model_tokenization_type(model)
    interaction = [{"role" : "user", "content" : "zero"}]
    with_gen = tokenizer.apply_chat_template(interaction, tokenize=True, add_generation_prompt=True)
    wo_gen = tokenizer.apply_chat_template(interaction, tokenize=True  , add_generation_prompt=False)

    if (eot_group == 3): return with_gen[-1:]
    elif (eot_group == 7) : return with_gen[-3:]
    else: return with_gen[len(wo_gen):]



def get_eot_suffix_tokens(tokenizer, model, role="user"):
    tokens = tokenizer.apply_chat_template([
        {"role" : "user", "content" : "zero"},
        {"role" : "assistant", "content" : "zero"},
        {"role" : "user", "content" : "x"}])
    _, eot_group, _ = model_tokenization_type(model)

    if   (eot_group == 1): return tokens[-1:]
    elif (eot_group == 2): return tokens[-2:]
    elif (eot_group == 3):
        if (role == 'assistant'): return tokens[-4:-3]
        else: return []
    elif (eot_group == 4):
        if (role == 'assistant'): return tokens[-3:-2]
        else: return []
    elif (eot_group == 5): return tokens[-7:-5]
    elif (eot_group == 6):
        if (role == 'user'): return tokens[-8:-7]
        else: return tokens[-9:-7]
    elif (eot_group == 7): return tokens[-5:-3]
    else:                  return []


def apply_chat_template_open(interaction, tokenizer, model):
    _, eot_group, _ = model_tokenization_type(model)
    tokens = tokenizer.apply_chat_template(
        interaction,
        add_generation_prompt=False
    )
    if (interaction[-1]["role"] == 'assistant'):
        if (eot_group == 2 or eot_group == 7): return tokens[:-2]
        elif (eot_group == 8): return tokens
        else: return tokens[:-1]
    else:
        if (eot_group == 2): return tokens[:-2]
        elif (eot_group == 8 or eot_group == 4): return tokens
        elif (eot_group == 7): return tokens[:-5]
        else: return tokens[:-1]


def apply_chat_template_closed(interaction, tokenizer, model):
    return apply_chat_template_open(interaction, tokenizer, model) + get_eot_suffix_tokens(tokenizer, model, role=interaction[-1]["role"])


def tokenize(s, tokenizer):
    return tokenizer.encode(s, add_special_tokens=False)


def tokenization_debugger(model=""):
    if (model == ""):
        model = os.environ.get("base_model")
    
    interaction1 = [
        {"role" : "user"     , "content" : "a"},
        {"role" : "assistant", "content" : "b"},
        {"role" : "user"     , "content" : "c"},
        {"role" : "assistant", "content" : "x"}
    ]

    tokenizer = AutoTokenizer.from_pretrained(model)

    print("="*140 + f"\n ---=||| MODEL NAME : {model} |||=--- \n" + "="*140)
    tokens = tokenizer.apply_chat_template(interaction1)

    assistant_prefix = get_assistant_prefix_tokens(tokenizer, model)
    user_prefix = get_user_prefix_tokens(tokenizer,model)
    assistant_eot = get_eot_suffix_tokens(tokenizer,model, role="assistant")
    user_eot = get_eot_suffix_tokens(tokenizer, model, role="user")

    print("STR-TOKENS:")
    print([tokenizer.decode(tok) for tok in tokens])
    print("\n\nASSISTANT PREFIX:")
    print([tokenizer.decode(tok) for tok in assistant_prefix])
    print("\nUSER PREFIX:")
    print([tokenizer.decode(tok) for tok in user_prefix])
    print("\nASSISTANT EOT:")
    print([tokenizer.decode(tok) for tok in assistant_eot])
    print("\nUSER EOT:")
    print([tokenizer.decode(tok) for tok in user_eot])
    print("\nSPLIT-PERSONALITY-TOKEN:")
    print([tokenizer.decode(tok) for tok in [get_hp_token_id(model)]])
    print ("=" * 140)


def apply_intervention(interaction, intervention, tokenizer, model, sp_type="hp", add_sp_token=False, return_mask=False):
    tokens = None
    mask   = None
    if  (sp_type  =="hp"):
        tokens = apply_chat_template_open(interaction, tokenizer, model)
        mask = [0.0]*len(tokens)

        if (add_sp_token):
            tokens += [get_hp_token_id(model)]
        tokens += tokenize(intervention, tokenizer)

    elif(sp_type =="up"): 
        tokens = apply_chat_template_closed(interaction, tokenizer, model) + get_user_prefix_tokens(tokenizer, model)
        mask = [0.0]*len(tokens)
        if (add_sp_token): tokens += [get_hp_token_id(model)]
        tokens += tokenize(intervention, tokenizer)
    
    elif (sp_type=="ur"):
        tokens = apply_chat_template_closed(interaction, tokenizer, model) + get_user_prefix_tokens(tokenizer, model)
        mask = [0.0]*len(tokens)
        tokens += tokenize(intervention, tokenizer)
        tokens += get_eot_suffix_tokens(tokenizer,model, role="user") + get_assistant_prefix_tokens(tokenizer,model)

        if (add_sp_token): tokens += [get_hp_token_id(model)]
    
    mask = mask + [1.]*(len(tokens) - len(mask))
    if (return_mask): return tokens, mask
    else:             return tokens


# ========================================================================================================================================================
# =========== MASKING + TOKENIZATION TEMPLATES ===========================================================================================================
# ========================================================================================================================================================

def claude_tokenizer_and_mask(interaction, I, R, F, tokenizer, model, elicitation_type="hp", add_sp_token=True):
    tot_tokens = apply_intervention(interaction, I, tokenizer, model, sp_type=elicitation_type, add_sp_token=add_sp_token)

    R_tokens   = tokenize(R, tokenizer)
    F_tokens   = tokenize(F, tokenizer)
    eot_tokens = get_eot_suffix_tokens(tokenizer, model, role=("assistant" if elicitation_type in ("hp","ur") else "user"))

    tot_mask = [0] * (len(tot_tokens) - 1) + [1] * (1 + len(R_tokens)) + [1] * len(F_tokens) + [0] * len(eot_tokens)
    tot_tokens = tot_tokens + R_tokens + F_tokens + eot_tokens
    return (tot_tokens, tot_mask)


def filter_by_length(samples, max_length):
    return [(tokens, mask) for tokens, mask in samples if len(tokens) <= max_length]

def pad(sample, length, tokenizer):
    tokens, mask = sample
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    tokens = t.cat([tokens, t.full((length - tokens.shape[0],), pad_token_id, dtype=t.long)], dim=0)
    mask   = t.cat([mask,   t.zeros(length - mask.shape[0], dtype=t.long)], dim=0)
    return (tokens, mask)




# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================