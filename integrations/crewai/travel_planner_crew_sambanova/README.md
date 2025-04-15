
<picture>
<a href="https://sambanova.ai/"\>
<source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
<img alt="SambaNova logo" src="../../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# Travel Planner

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

This is a demonstration of how a travel planner can be built using Agentic AI. Tools used include Gradio, CrewAI, and Sambanova Cloud.
You can test how this works at: https://huggingface.co/spaces/sambanovasystems/trip-planner.

## Installation using pip and venv
Ensure you have Python >=3.11 installed on your system.

Clone the start kit repo.

```
git clone https://github.com/sambanova/ai-starter-kit.git
```

Navigate to the *travel-planner-crew-sambanova* folder. Create a virtual environment. Then run pip install -r requirements.txt.

```bash
cd integrations/crewai/travel_planner_crew_sambanova
python3 -m venv ./
source ./venv/bin/activate
pip install -r requirements.txt
```

### API Key Setup 
**Before you begin, add your `SAMBANOVA_API_KEY` and `SERPER_API_KEY` into the `.env` file in the main starter kit repo folder.** Alternatively provide them as environment variables while running the app.
- You can get your SambaNova API Key [here](https://docs.astral.sh/uv/)
- You can get your Serper API Key [here](https://serper.dev/)

### Running
- Start the Gradio App by running:
```
python3 app.py
```
alternatively, if a .env file is not created...
```
SAMBANOVA_API_KEY=<Sambanova Key> SERPER_API_KEY=<Serper Key> python3 app.py
```

### Customizing
- Modify `config/agents.yaml` to define your agents. Please check CrewAI documentation on how to configure an agent.
- Modify `config/tasks.yaml` to define your tasks. Please check CrewAI documentation on how to configure a task.
