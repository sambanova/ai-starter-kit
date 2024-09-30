//
var api_key = 0;
var url = 'https://api.sambanova.ai/v1/chat/completions';

/**
 * Event trigger that creates the custom menu
*/
function onOpen() {
  var ui = DocumentApp.getUi();
  ui.createMenu('🟠 SambaAI')
      .addItem('Clean grammar', 'clean_grammar')
      .addItem('Rephrase', 'professionalize')
      .addToUi();
}

/**
 * Clean grammar option, which replaces highlighted text with
 * Llama3 405B generated text with proper grammar
*/
function clean_grammar() {
  var body = DocumentApp.getActiveDocument().getSelection();
  Logger.log(body.getRangeElements());
  body.getRangeElements().forEach(e => {
    const start = e.getStartOffset();
    const end = e.getEndOffsetInclusive();
    const prompt = e.getElement().asText();
    if (start == -1 && end == -1) { // means highlighted text encapsulates the whole element
      var input = prompt.getText();
    } else {
      var input = prompt.getText().substring(start, end + 1);
    }
    Logger.log(input);
    if (input.length == 0) {  // means its an empty line
      return;
    }
    var resp = make_api_call("Minimally change this text to have good grammar. Don't explain the changes.", input);
    if (start == -1 && end == -1) {
      prompt.setText(resp);
    } else {
      prompt.deleteText(start, end);
      prompt.insertText(start, resp);
    }
  });
}

/**
 * Rephrase option, which appends Llama3 405B generated text to the document
*/
function professionalize() {
  var input = get_input();
  var resp = make_api_call("Rephrase to have a professional tone, using concise and powerful language.", input);
  DocumentApp.getActiveDocument().appendParagraph("SambaAI rephrase: " + resp);
}

function get_input(){
  var body = DocumentApp.getActiveDocument().getSelection().getRangeElements();

  if (body) {
    var input = ""
    body.forEach(i => {
      input += i.getElement().asText().getText();
    })
  } else {
    var input = DocumentApp.getActiveDocument().getActiveTab().asDocumentTab().getBody().getText();
  } 
  Logger.log(input);
  return input;
}

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
    "method" : "post",
    "headers" : {
      "Authorization": "Bearer " + api_key,
      "Content-Type": "application/json"
      },
    "payload" : JSON.stringify(data)
  };

  var response = UrlFetchApp.fetch(url, options).getContentText();
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