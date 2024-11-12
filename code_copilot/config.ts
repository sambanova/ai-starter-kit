//SambaNova Cloud usage
const sambanovacloud_api_key = "your-SambaNovaCloud-api-key";
const sambanovacloud_model = "Meta-Llama-3.1-70B-Instruct";

//SambaStudio usage
const sambastudio_base_url = "your-sambastudio-base-url";
const sambastudio_project_id = "your-sambastudio-project-id";
const sambastudio_endpoint_id = "your-sambastudio-endpoint-id";
const sambastudio_api_key = "your-sambastudio-api-key";
const sambastudio_use_bundle = true;
const sambastudio_model = "Meta-Llama-3-70B-Instruct-4096"; //when using bundle endpoint

/**
 * Llama3 template structure
 * 
 * Generates a template for Llama3 messages.
 * 
 * @param {ChatMessage[]} msgs - An array of chat messages.
 * @returns {string} A formatted prompt string for Llama3.
 */
function templateLlama3Messages(msgs: ChatMessage[]): string {
  let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.\n";
  if (msgs[0].role === "system") {
    prompt += `${msgs[0].content}\n`;
    msgs.shift();
  }
  prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>"
  prompt += "Instruction:\n";
  for (let msg of msgs) {
    prompt += `${msg.content}\n`;
  }
  prompt += "Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|> \n";
  return prompt;
}

/**
 * sambaStudioEndpoint handler
 * 
 * This function is an asynchronous generator that handles Streaming API calls to a SambaNova endpoint.
 * 
 * The function sends a POST request to the specified `url` with the provided `body` and `extraHeaders`.
 * The function then reads the response body as a stream and decodes it as text.
 * It splits the text into lines and yields each line as a JSON object.
 * 
 * The yielded objects contain a `result` property with a `responses` property, which contains a `stream_token` property.
 * 
 * @param {string} url - The URL of the SambaNova endpoint
 * @param {string} key - The API key
 * @param {string} body - The request body
 * @param {Object} [extraHeaders] - Additional headers to include in the request
 * @returns {IterableIterator<string>} An iterable iterator yielding stream tokens
 */
async function* sambaStudioEndpointHandler(
    url: string,
    key: string, 
    body: string, 
    extraHeaders: { [key: string]: string } | null = null
  ) {
    let headers = {
      'Content-Type': 'application/json',
      'key': key,
    };
    if (extraHeaders !== null) {
      headers = Object.assign({}, headers, extraHeaders);
    }
    const response = await fetch(url, {
      method: 'POST',
      headers: headers,
      body: body,
    });
    if (!response.ok) {
      const responseBody = await response.text();
      console.error('API call failed with status:', response.status);
      console.error('Response body:', responseBody);
      throw new Error(`API call failed with status ${response.status}`);
    }
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get reader from response body');
    }
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let lines = buffer.split('\n');
      buffer = lines.pop()!;
      for (const line of lines) {
        if (line.trim()) {
          try {
            const json_line = JSON.parse(line);
            yield json_line.result.items[0].value.stream_token;
          } catch (error) {
            console.error('failed to parse JSON: ', line)
          }
        }
      }
    }
  }

/**
 * SambaStudio model
 */
let SambaStudioModel = {
  options: {
    title: "SambaStudio",
    model: sambastudio_model,
    contextLength: 4096,
    templateMessages: templateLlama3Messages,
  },
  /**
   * Stream completion function for SambaStudio
   * 
   * @param {string} prompt - The prompt to generate text for
   * @param {CompletionOptions} options - Options for the completion
   */
  streamCompletion: async function* (
    prompt: string,
    options: CompletionOptions,
  ) {
    const url = `${sambastudio_base_url}/api/v2/predict/generic/stream/${sambastudio_project_id}/${sambastudio_endpoint_id}`;
    let body = ""
    if(sambastudio_use_bundle){
      body = JSON.stringify({
        items:[{"id": "item0", "value": prompt}],
        params: {
          select_expert: options.model,
          do_sample: true,
          process_prompt: false,
          max_tokens_to_generate: 1024,
          temperature:"0.7",
        }
      });
    }
    else{
      body = JSON.stringify({
        instance: prompt,
        params: {
          do_sample:true,
          process_prompt: false,
          max_tokens_to_generate:1024,
          temperature:"0.7",
        }
      });
    }
    yield* sambaStudioEndpointHandler(url, sambastudio_api_key, body);
  }
};


/**
 * sambaNovaCloudEndpoint handler
 * 
 * This function is an asynchronous generator that handles Streaming API calls to a SambaNova endpoint.
 * 
 * The function sends a POST request to the specified `url` with the provided `body` and `extraHeaders`.
 * The function then reads the response body as a stream and decodes it as text.
 * It splits the text into lines and yields each line as a JSON object.
 * 
 * The yielded objects contain a `result` property with a `responses` property, which contains a `stream_token` property.
 * 
 * @param {string} url - The URL of the SambaNova endpoint
 * @param {string} key - The API key
 * @param {string} body - The request body
 * @param {Object} [extraHeaders] - Additional headers to include in the request
 * @returns {IterableIterator<string>} An iterable iterator yielding stream tokens
 */
async function* sambaNovaCloudEndpointHandler(
  url: string,
  key: string, 
  body: string, 
  extraHeaders: { [key: string]: string } | null = null
) {
  let headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${key}`,
  };
  if (extraHeaders !== null) {
    headers = Object.assign({}, headers, extraHeaders);
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: headers,
    body: body,
  });

  if (!response.ok) {
    const responseBody = await response.text();
    console.error('API call failed with status:', response.status);
    console.error('Response body:', responseBody);
    throw new Error(`API call failed with status ${response.status}`);
  }
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Failed to get reader from response body');
  }
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let lines = buffer.split('\n');
    buffer = lines.pop()!;
    for (const line of lines) {
      let str_line = line.trim();
      if (str_line) {
        if (str_line === "data: [DONE]"){
          break;
        }
        else {
          str_line = str_line.replace(/^data: /, '')
          try {
            const json_line = JSON.parse(str_line);
            yield json_line.choices[0].delta.content;
          } catch (error) {
            console.error('failed to parse JSON: ', line)
          }
        }
      }
    }
  }
}

/**
 * SambaNova Cloud model
 */
let SambaNovaCloudModel = {
  options: {
    title: "SambaNovaCloud",
    model: sambanovacloud_model,
    contextLength: 4096,
    templateMessages: templateLlama3Messages,
  },
  /**
   * Stream completion function for SambaNovaCloud
   * 
   * @param {string} prompt - The prompt to generate text for
   * @param {CompletionOptions} options - Options for the completion
   */
  streamCompletion: async function* (
    prompt: string,
    options: CompletionOptions,
  ) {
    const url = 'https://api.sambanova.ai/v1/chat/completions';
    const extraHeaders = {
    };
    const body = JSON.stringify({
      messages:[  
        {  
            role: "user", 
            content: prompt
        }
      ], 
      model: options.model,
      stop: ["<|eot_id|>"],
      stream: true,
      max_tokens: 1024,
    });
    yield* sambaNovaCloudEndpointHandler(url, sambanovacloud_api_key, body, extraHeaders);
  },
};

/**
 * Modifies the given configuration by adding SambaStudioModel and SambaNovaCloudModel to the models array.
 * 
 * @param {Config} config - The configuration object to be modified.
 * @returns {Config} The modified configuration object.
 */
export function modifyConfig(config: Config): Config {
  config.models.push(SambaStudioModel)
  config.models.push(SambaNovaCloudModel)
  //config.tabAutocompleteModel=SambaStudioModel;
  //config.tabAutocompleteModel=SambaNovaCloudModel;
  return config; 
}