//Sambaverse usage
const sambaverse_api_key = "your-sambaverse-api-key";

//SambaStudio usage
const sambastudio_base_url = "your-sambastudio-base-url";
const sambastudio_project_id = "your-sambastudio-project-id";
const sambastudio_endpoint_id = "your-sambastudio-endpoint-id";
const sambastudio_api_key = "your-sambastudio-api-key";

// Llama3 template structure
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

// SambaNova endpoint handler
async function* endpointHandler(
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
            yield json_line.result.responses[0].stream_token;
          } catch (error) {
            console.error('failed to parse JSON: ', line)
          }
        }
      }
    }
  }

// SambaStudio model
let SambaStudioModel = {
  options: {
    title: "SambaStudio",
    model: "Meta-Llama-3-8B-Instruct",
    contextLength: 4096,
    templateMessages: templateLlama3Messages,
  },
  streamCompletion: async function* (
    prompt: string,
    options: CompletionOptions,
  ) {
    const url = `${sambastudio_base_url}/api/predict/generic/stream/${sambastudio_project_id}/${sambastudio_endpoint_id}`;
    const body = JSON.stringify({
      instance: prompt,
      params: {
        select_expert: {
          type: "string",
          value: options.model
        },
        process_prompt: {
          type: "bool",
          value: "false"
        },
        max_tokens_to_generate: {
          type: "int",
          value: "1024"
        }
      }
    });
    yield* endpointHandler(url, sambastudio_api_key, body);
  }
};

// Sambaverse model
let SambaverseModel = {
  options: {
    title: "Sambaverse",
    model: "Meta-Llama-3-8B-Instruct",
    contextLength: 4096,
    templateMessages: templateLlama3Messages,
  },
  streamCompletion: async function* (
    prompt: string,
    options: CompletionOptions,
  ) {
    const url = 'https://sambaverse.sambanova.ai/api/predict';
    const extraHeaders = {
      "modelName": "Meta/Meta-Llama-3-8B-Instruct"
    };
    const body = JSON.stringify({
      instance: prompt,
      params: {
        select_expert: {
          type: "string",
          value: options.model
        },
        process_prompt: {
          type: "bool",
          value: "false"
        },
        max_tokens_to_generate: {
          type: "int",
          value: "1024"
        }
      }
    });
    yield* endpointHandler(url, sambaverse_api_key, body, extraHeaders);
  },
};

export function modifyConfig(config: Config): Config {
  config.models.push(SambaStudioModel)
  config.models.push(SambaverseModel)
  //config.tabAutocompleteModel=SambaStudioModel;
  //config.tabAutocompleteModel=SambaverseModel;
  return config;
}