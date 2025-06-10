async def prompt_tester(input_object, prompt_beg, prompt_end, pretty_format=False, printt=True, input_limit=50, static_random=False):
    inp_tokens = 0
    out_tokens = 0
    prompts = {}

    if static_random:
        np.random.seed(0)

    input_object = {k: v if len(v) <= input_limit
                    else [v[i] for i in
                          np.random.choice(
                    len(v), 
                    size=input_limit, 
                    replace=False  # No duplicates
                )]
                for k, v in input_object.items()}
    # return input_object

    
    for k, v in input_object.items():
        if pretty_format:
            formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(v)])
        else:
            formatted_texts = json.dumps(v, indent=4, ensure_ascii=False)
        prompts[k] = prompt_beg+formatted_texts+prompt_end
        # print(prompts[k])
        # break
    
    tasks = []
    for k, prompt in prompts.items():
        tasks.append(agenerate_text(prompt, return_token_usage=True))
        # print(k, prompt)
        # break

    results = await asyncio.gather(*tasks)
    results = {k: result for k, result in zip(prompts.keys(), results)}

    output = {}
    for k, v in results.items():
        # if print_token_usage:
        v, prompt_tokens, candidate_tokens = v
        inp_tokens += prompt_tokens
        out_tokens += candidate_tokens
        if printt:
            print(k, prompts[k])
            print("-")
            print(k, v)
            print("-")
        print(inp_tokens, out_tokens)
        print("------------------")

        output[k] = v
    
    return output
    

def prompt_json_output_cleaner(output):
    out = {}

    for k, v in output.items():
        out_match = re.search(r"```json\s*\n(.*?)\n\s*```", v, re.DOTALL | re.IGNORECASE)
        
        try:
            if out_match:
                # output.append(json.loads(out_match.group(1).strip()))
                outt = json.loads(out_match.group(1).strip())
        except Exception as e:
            print(k, "error", e)
            # print(k, v)
            continue
        # print(k, v)
        
        if len(outt) != 0:
            out[str(k)] = outt
    
    return out