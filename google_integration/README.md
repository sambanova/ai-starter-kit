
<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

SambaCloud-Google-Integration
===============================

App Scripts intended for those with SambaCloud API keys to integrate LLMs into Google Workspaces

Getting Started:
1. Sign up for an API key at <a href="https://cloud.sambanova.ai/apis">SambaNova Cloud</a>. It's free and will give you access to the Llama 3 405B model.
2. Open your document/sheet and select the menu item Extension > App Script
3. Copy the respective JS script for your Google application into the App Script window
4. Replace the 0 in `api_key = 0` with the API key in your email.


## Google Documents Features
Highlight a piece of text, then go to the “SambaAI” menu item and select `Clean grammar`. This will fix any grammar issues in place.
`Rephrase` will append a new paragraph with a concise and professional rephrasing of the selected text.

## Google Sheets Features
Type the custom function `=SAMBA_AI_QUESTION()` for general Llama3 use and `=SAMBA_AI_FORMULA()` when requesting for a specific formula from the model. Credit to <a href="https://www.sheetai.app/">SheetAI</a> for the idea.

Example use cases include the following: \
`=SAMBA_AI_FORMULA("take B1 if A1 is true, but otherwise take C1, and add that result to D1")` \
`=SAMBA_AI_QUESTION("What is the RSVP of this person: yes, no, or maybe? she said "&A10)` \
`=SAMBA_AI_QUESTION("Which essay is more compelling? Prompt 1:"&A13&" Prompt 2:"&B13)`