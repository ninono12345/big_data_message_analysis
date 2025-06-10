import os
# service account embeddingams
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "..."

print(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))



gemini_api = "..."

import os
# from google import genai as genai2
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
# from google.genai.types import HttpOptions, Part

genai.configure(api_key=gemini_api)



generation_config = {
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8000,
  "response_mime_type": "text/plain",
}

model_name = "gemini-2.0-flash"

model_global = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
    system_instruction="You are a helpful assistant.",
)

safety_settings={
  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

m = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config
)

# token_client = genai2.Client(http_options=HttpOptions(api_version="v1"))


async def agenerate_text(prompt, system_prompt=None, return_token_usage=False, print_token_usage=False, history_messages=[], jsonn=False, **kwargs):
    try:
        # if print_token_usage:
        #     contents = [prompt]
        #     response1 = token_client.models.count_tokens(model=model_name, contents=contents)
        #     print("tokens1", response1)

        
        # chat_session = m.start_chat(history=[])
        # response = await chat_session.send_message_async(prompt, safety_settings=safety_settings)
        response = await m.generate_content_async(prompt, safety_settings=safety_settings)
        if print_token_usage:
            print("tokens2", response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

        if return_token_usage:
            return response.text, response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count
        return response.text
    except Exception as e:
        print("ERROR ERROR in agenerate_text", e)
        if return_token_usage:
            return "", 0, 0
        return ""



PROJECT_ID = "primal-buttress-461813-s2"
MODEL_ID = "text-multilingual-embedding-002"
REGION = "us-central1"

import vertexai
# import vertexai.generative_models
# import vertexai.generative_models._prompts
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

vertexai.init(project=PROJECT_ID)

model_vertex = TextEmbeddingModel.from_pretrained(MODEL_ID)

# model_response = gemini_model.count_tokens([...])



async def aembed_text(
    texts: list = None,
    task: str = "CLUSTERING",
    dimensionality = 768,
):
    """Embeds texts with a pre-trained, foundational model.
    Args:
        texts (List[str]): A list of texts to be embedded.
        task (str): The task type for embedding. Check the available tasks in the model's documentation.
        dimensionality (Optional[int]): The dimensionality of the output embeddings.
    Returns:
        List[List[float]]: A list of lists containing the embedding vectors for each input text
    """
    # model = TextEmbeddingModel.from_pretrained(MODEL_ID)
    model = model_vertex
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = await model.get_embeddings_async(inputs, **kwargs)
    ret = [embedding.values for embedding in embeddings]
    # convert to ndarray
    return np.array(ret)

# naudojam sita pavieniams
async def aembed_text_wrapper(texts, task = "CLUSTERING", dimensionality = 768):
    print("starting embedding")
    # return np.array(await aembed_text(texts, task, dimensionality))
    try:
        return await aembed_text(texts, task, dimensionality)
    except Exception as e:
        print("error in aembed_text_wrapper", e)
        return None

# input id->cleaned_text output: id->embedding
async def aembed_many_texts(textsd: dict[str, str], gotten: list[str]=None, taskk = "CLUSTERING", dimensionality = 768, batch_size = 20) -> dict[str, np.ndarray]:
    ids = list(textsd.keys())
    texts = list(textsd.values())
    if gotten is not None:
        to_get = set(textsd) - set(gotten)
        textsd = {k: textsd[k] for k in to_get}
    tasks = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    task_id_tracker = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
    print(len(tasks))
    tasks = [task for task in tasks if len(task) > 0]

    embeddings = await asyncio.gather(*[aembed_text_wrapper(task, taskk, dimensionality) for task in tasks])
    ret = {}
    # check if embeddings are not None
    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            for j, emb in enumerate(embedding):
                ret[task_id_tracker[i][j]] = emb

    return ret

# input cleaned_text output: embedding (ndarray)
async def aembed_many_texts_np(texts: list[str], taskk = "CLUSTERING", dimensionality = 768, batch_size = 20) -> dict[str, np.ndarray]:
    # ids = list(texts.keys())
    # texts = list(texts.values())
    tasks = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    # task_id_tracker = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
    print(len(tasks))
    tasks = [task for task in tasks if len(task) > 0]

    embeddings = await asyncio.gather(*[aembed_text_wrapper(task, taskk, dimensionality) for task in tasks])
    return np.concatenate(embeddings, axis=0)
