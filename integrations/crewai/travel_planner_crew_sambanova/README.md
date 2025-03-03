
<picture>
<a href="https://sambanova.ai/"\>
<source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
<img alt="SambaNova logo" src="../../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

#Travel Planner


Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

This is a demonstration of how a travel planner can be built using Agentic AI. Tools used include Gradio, Crew AI, and the backend supported by Sambanova Cloud.
You can test how this works at: https://huggingface.co/spaces/sambanovasystems/trip-planner.

## Installation
Ensure you have Python >=3.10 <=3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

Clone the start kit repo.

```
git clone https://github.com/sambanova/ai-starter-kit.git
```

Navigate to the *travel-planner-crew-sambanova* folder. Then run pip install -r requirements.txt:

```bash
cd integrations/crewai/travel_planner_crew_sambanova
```

```bash
pip install uv crewai 
```

Next, install the dependencies (this will create a .venv automatically via the crewai command line):
(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

Next, activiate the .venv in the project repo that was created by the previous step, you should see an edu-flow env activated:
```bash
source .venv/bin/activate


### API Key Setup 
**Before you begin Add your `SAMBANOVA_API_KEY` and `SERPER_API_KEY` into the `.env` file in the main starter kit repo folder. Alternatively provide them as environment variables.**
- You can get your SambaNova API Key [here](https://docs.astral.sh/uv/)
- You can get your Serper API Key [here](https://serper.dev/)

### Running
- Start the Gradio App by running python3 app.py

### Customizing
- Modify `src/edu_flow/config/agents.yaml` to define your agents
