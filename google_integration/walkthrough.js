var api_key = 0; // replace with personal API key
var url = 'https://api.sambanova.ai/v1/chat/completions';

/**
 * Ask Sambanova Cloud anything that plagues your mind
 * @param {string} input The prompt
 * @return The response
 * @customfunction
*/
function SAMBA_AI_QUESTION(input) {
    var system_input = "Answer in a short phrase.";

    // API payload
    var data = {
        "messages": [
        {"role": "system", "content": system_input},
        {"role": "user", "content": input}
        ],
        "stop": ["<|eot_id|>"],
        "model": "Meta-Llama-3.1-405B-Instruct",
        "stream": true, "stream_options": {"include_usage": true}
    }

    var options = {
    "method" : "post",
    "headers" : {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
        },
    "payload" : JSON.stringify(data)
    };

    // Get api endpoint
    var response = UrlFetchApp.fetch(url, options).getContentText(); 
    // Logger.log(response);    // uncomment to see the response format

    // Data is received as a string of few tokens at a time that must be parsed
    response = response.split('data: ')
    // Last response will always be `[DONE]` which can be removed
    response.pop()

    var resp_paragraph = ""
    response.forEach(i => {
    if (i != "") {
        json = JSON.parse(i)
        if (json?.choices[0]?.delta?.content)
            resp_paragraph += json.choices[0].delta.content
    }
    })
    return resp_paragraph
}