<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>

# SambaNova Financial Agent using CrewAI (Financial Flow)
======================

Questions? Just <a href="https://discord.gg/54bNAqRw" target="_blank">message us</a> on Discord <a href="https://discord.gg/54bNAqRw" target="_blank"><img src="https://github.com/sambanova/ai-starter-kit/assets/150964187/aef53b52-1dc0-4cbf-a3be-55048675f583" alt="Discord" width="22"/></a> or <a href="https://github.com/sambanova/ai-starter-kit/issues/new/choose" target="_blank">create an issue</a> in GitHub. We're happy to help live!

Welcome to the Financial Flow project, powered by [crewAI](https://crewai.com)!

Table of Contents:

- [1. Overview](#overview)
- [2. Setup](#setup)
- [3. Installation](#installation)
  - [3.1 Using UV and the CrewAI commands](#uv-crewai-commands)
  - [3.2 Using pip and venv](#pip-venv)
  - [3.3 The Streamlit app](#streamlit)
- [4. Customization](#customization)
- [5. Understanding and monitoring your crews](#understanding-monitoring)

## 1. Overview

The Financial Flow project is a comprehensive template designed to facilitate the setup of a multi-agent AI system.
By leveraging the robust and adaptable framework offered by crewAI,
this project aims to enhance collaboration among AI agents to perform complex financial tasks efficiently.
The ultimate goal of the Financial Flow is to maximize collective intelligence and capabilities to produce high-quality financial reports.

This app demonstrates the capabilities of large language models (LLMs)
in extracting and analyzing financial data using function calling, web scraping,
and retrieval-augmented generation (RAG).

Use the navigation menu to explore various features including:

-- **Generic Google Search**: Scrape the web using
  <a href="https://serper.dev/" target="_blank">Serper</a> Google Search API.
- **Stock Data Analysis**: Query and analyze stocks based on
  <a href="https://pypi.org/project/yfinance/" target="_blank">Yahoo Finance</a> data.
- **Financial Filings Analysis**: Query and analyze financial filings based on 
  <a href="https://www.sec.gov/edgar/search/" target="_blank">SEC EDGAR </a> data.
- **Financial News Scraping**: Scrape financial news articles from 
  <a href="https://uk.finance.yahoo.com/" target="_blank">Yahoo Finance</a> News.

## 2. Setup

1. Add your `SAMBANOVA_API_KEY` to the `.env` file.

2. Add your `SERPER_API_KEY` to the `.env` file.

3. For the `SEC-EDGAR` functionalities, company name and email are used to form a user-agent of the form:
  USER_AGENT: ```<Company Name> <Email Address>```.

  Add the following to the `.env` file:
  ```
  # Your organization
  SEC_API_ORGANIZATION="<your organization>"

  # Your email address
  SEC_API_EMAIL="<user@email_provider.com>"
  ```
    
## 3. Installation

Ensure you have Python `>=3.11 <3.13` installed on your system.
All the packages/tools are listed in the `requirements.txt` file in the project root directory.

If you want to create a Python virtual environment with its built-in module `venv`
and then install the dependencies using `pip`,
follow the steps below.

1. Install and update `pip`.

```bash
cd ai-starter-kit/financial_agent_crewai
python3 -m venv financial_agent_crewai
source financial_agent_crewai/bin/activate
pip install -r requirements.txt
```

2. Run the `main.py` file:

```bash
cd ai-starter-kit/financial_agent_crewai
python main.py
```

This command initializes the `FinancialFLow` Flow, assembling the agents and assigning them tasks as defined in your configuration.
This example, unmodified, will generate a `report.md` file and a `report.pdf` file
as the outputs of a financial research and analyais in the `cache` folder.

### 3.3 The `Streamlit` app
After building your virtual environment, either using `uv` or using `pip`,
you can run our `streamlit` app for an interactive interface and monitoring.

Run the following command:

```bash
streamlit run streamlit/streamlit_app.py --browser.gatherUsageStats false 
```
or, if Streamlit does not recognize your virtual environment due to a path mismatch, run the following command:

```bash
python -m streamlit run streamlit/streamlit_app.py --browser.gatherUsageStats false 
```

You can now enter your query and select which data sources you want to use among the following:
1. Generic Google Search.
2. SEC EDGAR Filings.
3. Yahoo Finance News.
4. Yahoo Finance Stocks.

## 4. Customization
You can modify the following hyperparameters of the flow and of the crews,
as well as the corresponding agents and tasks of the latter,
in the `config.py` file.

- Level of verbosity
  `VERBOSE`

- Specify the directory and file path of the `cache`
  `CACHE_DIR`

- Maximum number of words per section
  `MAX_SECTION_WORDS`

- The default data sources to use in case they are not given to the `FinancialFlow` as hyperparameters.
  1. Generic Google Search: `SOURCE_GENERIC_SEARCH`.
  2. SEC EDGAR Filings: `SOURCE_SEC_FILINGS`.
  3. Yahoo Finance News: `SOURCE_YFINANCE_NEWS`.
  4. Yahoo Finance News: `SOURCE_YFINANCE_STOCK`.

- User query:
  `USER_QUERY`

- Number of documents to use for RAG
  `NUM_RAG_SOURCES`

- Maximum number of urls in generic Google web search or Yahoo Finance web search
  `MAX_NEWS`

- Maximum news per ticker symbol
  `MAX_NEWS_PER_TICKER`

- LLMs by crew

- LLM temperature
  `TEMPERATURE`

## 5. Understanding and monitoring your crews

The `FinancialFlow`, defined in `main.py`, is a `crewai.flow.flow.Flow` pipeline orchestrating several crews.

The `FinancialFlow` consists of multiple crews, each one composed of AI agents, each with unique roles, goals, and tools.
The `config/agents.yaml` files outline the capabilities and configurations of each agent in your crew.
These agents collaborate on a series of tasks, defined in the `config/tasks.yaml` files,
leveraging their collective skills to achieve complex objectives.
```