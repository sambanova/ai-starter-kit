<picture>
<a href="https://sambanova.ai/"\>
<source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
<img alt="SambaNova logo" src="../../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# Educational Research Crew

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!
Welcome to the Edu-Research Crew project, a powerful collaboration between [crewAI](https://crewai.com) and [SambaNova Systems](https://sambanova.ai). This template demonstrates how to create sophisticated multi-agent AI systems for educational research, powered by SambaNova's state-of-the-art LLMs and crewAI's innovative framework. Our partnership aims to revolutionize educational content creation and research capabilities by combining SambaNova's powerful language models with crewAI's multi-agent orchestration.

## Installation
Ensure you have Python >=3.10 <=3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.


Clone the start kit repo.

```
git clone https://github.com/sambanova/ai-starter-kit.git
```


First, navigate to the *edu-planner-crew-sambanova* folder and pip install uv and crewai:

```bash
cd integrations/crewai/edu_planner_crew_sambanova
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
```


### API Key Setup 
**Before you begin Add your `SAMBANOVA_API_KEY` and `SERPER_API_KEY` into the `.env` file in the main starter kit repo folder**
- You can get your SambaNova API Key [here](https://docs.astral.sh/uv/)
- You can get your Serper API Key [here](https://serper.dev/)

### Customizing
- Modify `src/edu_flow/config/agents.yaml` to define your agents
- Modify `src/edu_flow/config/tasks.yaml` to define your tasks
- Modify `src/edu_flow/crew.py` to add your own logic, tools and specific args
- Modify `src/edu_flow/main.py` to add custom inputs for your agents and tasks

## Running the Project
You can run the project in two ways:
1. Command Line Interface (change config.py to set parameters):
```bash
crewai flow kickoff
```
2. Streamlit Interface (preferred):
```bash
streamlit run src/edu_flow/streamlit_app.py
```
Both methods will initialize the edu-flow Crew, assembling the SambaNova-powered agents and assigning them tasks as defined in your configuration.
This example, unmodified, will create a `report.md` file with the output of educational research in the root folder, leveraging SambaNova's advanced LLMs for comprehensive analysis.

## Understanding Your Crew
The edu-flow Crew combines SambaNova's powerful language models with crewAI's multi-agent framework to create an intelligent system specifically designed for educational research and content creation. Each agent, powered by SambaNova's LLMs, has unique roles, goals, and tools defined in `config/agents.yaml`. These agents collaborate on tasks outlined in `config/tasks.yaml`, working together to achieve complex educational objectives.

## Support
For support, questions, or feedback regarding the Edu-Research Crew:
- Visit the [crewAI documentation](https://docs.crewai.com)
- Check out [SambaNova's AI solutions](https://sambanova.ai)
- Reach out through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)
Experience the future of educational AI with the combined power of crewAI and SambaNova Systems.