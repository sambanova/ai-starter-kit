var api_key = 0; // replace with personal API key
var url = 'https://api.sambanova.ai/v1/chat/completions';


/**
 * Helper function to make the API call
 * @param {string} input User prompt
 * @param {string} context System prompt
 * @return The response
 * @customfunction
*/
function make_api_call(context, input) {
    var data = {
        "messages": [
        {"role": "system", "content": context},
        {"role": "user", "content": input}
        ],
        "stop": ["<|eot_id|>"],
        "model": "Meta-Llama-3.1-405B-Instruct",
        "stream": true, "stream_options": {"include_usage": true}
    }

    var options = {
        // "async": true,
        // "crossDomain": true,
        "method" : "post",
        // "muteHttpExceptions": true,
        "headers" : {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
        },
        "payload" : JSON.stringify(data)
    };

    var response = UrlFetchApp.fetch(url, options).getContentText(); // get api endpoint
    response = response.split('data: ')
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

/**
 * Ask Sambanova Cloud for a specific formula
 * @param {string} input The prompt
 * @return The formula
 * @customfunction
*/
function SAMBA_AI_FORMULA(input) {
    make_api_call("Answer the question with only a Google Sheets formula.", input);
}

/**
 * Ask Sambanova Cloud anything that plagues your mind
 * @param {string} input The prompt
 * @return The response
 * @customfunction
*/
function SAMBA_AI_QUESTION(input)  {
    make_api_call("Answer in a short phrase.", input);
}
