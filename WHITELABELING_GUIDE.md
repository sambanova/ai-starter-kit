# White Labeling and Customization Guide

## SambaNova AI Starter Kits

---

## Table of Contents

- [Overview](#overview)
- [Section 1: Getting your AI Starter Kits Fork](#section-1-getting-your-ai-starter-kits-fork)
- [Section 2: Documentation Customization](#section-2-documentation-customization)
- [Section 3: AI Starter Kits UI Customization](#section-3-ai-starter-kits-ui-customization)
- [Section 4: Notebooks and Quickstarts Customization](#section-4-notebooks-and-quickstarts-customization)
- [Section 5: Recommended White Labeling Tools and Automation](#section-5-recommended-white-labeling-tools-and-automation)
- [Section 6: Merging Changes to Your Main](#section-6-merging-changes-to-your-main)
- [Section 7: Deployment Overview](#section-7-deployment-overview)
- [Additional Documentation and Resources](#additional-documentation-and-resources)
- [Document Changelog](#document-changelog)
- [License & Attribution](#license--attribution)

---

## Overview

This guide provides comprehensive instructions for customizing and white labeling SambaNova's AI Starter Kits and for your deployment. As a SambaManaged or SambaStack customer with your own branded API infrastructure, you can adapt these resources to:

- Replace SambaNova branding with your company's branding
- Configure applications to use your custom platform URL (e.g., `cloud.custx.ai`)
- Customize documentation, codes, and user interfaces with your branding

### Prerequisites

- Git installed and configured
- Python 3.10+ installed
- Access to your white-labeled infrastructure
- API keys for your custom platform
- Basic familiarity with:
  - Git and GitHub workflows
  - Environment variables and configuration files
  - Text editors or IDEs
  - Command line interface
  - Docker
  - Python scripts
  - Jupyter Notebooks
  - Streamlit applications

---

## Section 1: Getting your AI Starter Kits Fork

- Fork the [SambaNova ai-starter-kit repository](https://github.com/sambanova/ai-starter-kit/)

    > unselect the checkbox `only copy main branch`

- Clone your new forked repository. For example:

```bash
    git clone https://github.com/<custx_org>/ai-starter-kit/
```

- Checkout to a new branch from the white labeling branch. For example:

```bash
    git fetch origin sambamanaged
    git checkout -b whitelabel-customization origin/sambamanaged
```

## Section 2: Documentation Customization

### 2.1 Documentation Sources and Structure

The AI Starter Kits documentation is organized as follows:

**Main documentation files**:

```
ai-starter-kit/
├── README.md                                   # Root README
├── CONTRIBUTING.md                             # Contribution guidelines
├── images/                                     # Logo and icon assets
│   ├── dark-logo.png                           # Main dark theme logo
│   ├── light-logo.png                          # Main light theme logo
│   └── icon.svg                                # Icon/favicon
│   └── ...
└── [kit_name]/
    └── README.md                               # Kit-specific README
    └── streamlit/                              
        └── app_description.yaml                # Kit-specific UI description
```

**Kit-specific READMEs**:

Each starter kit contains its own README:

- `enterprise_knowledge_retriever/README.md`
- `multimodal_knowledge_retriever/README.md`
- `function_calling/README.md`
- `search_assistant/README.md`
- `benchmarking/README.md`
- `financial_assistant/README.md`
- `document_comparison/README.md`
- `data_extraction/README.md`
- `eval_jumpstart/README.md`
- `chat_templates/README.md`

### 2.2 Documentation Tooling

**Format**: Standard Markdown (.md files)

**No specialized generators**: The documentation uses GitHub-flavored Markdown without external documentation generators like MkDocs or Sphinx.

This means customization is straightforward:
- Edit files directly with any text editor or IDE (VSCode recommended) 
- Preview in GitHub or using Markdown preview tools
- No build process required for documentation

### 2.3 Customization Process

> See Section 5 Recommended White Labeling Tools and Automation to see bulk edition suggestions.

#### Step 1: Identify the Branding Elements to Replace

All READMEs contain the following SambaNova-specific elements:

1. **Logo** (at the top of each README):
```markdown
<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/light-logo.png" height="100">
  <img alt="SambaNova logo" src="./images/dark-logo-.png" height="100">
</picture>
</a>
```

2. **Platform URL**: `https://cloud.sambanova.ai`
3. **Community URL**: `https://community.sambanova.ai`
4. **Company Website**: `https://sambanova.ai/`
5. **GitHub Repository**: `https://github.com/sambanova/ai-starter-kit`

#### Step 2: Replace the Logo Files

Place your company's logo files in the `images/` directory:

```bash
# Add your logo files (replace <your-dark-logo>, <your-light-logo>, <your-icon> with your actual filenames; keep the same target filenames for easier migration)
cp /path/to/<your-dark-logo>.png images/dark-logo.png
cp /path/to/<your-light-logo>.png images/light-logo.png
cp /path/to/<your-icon>.svg images/icon.svg
```

**Logo specifications**:

- **Dark Logo**: Used on light backgrounds
- **Light Logo**: Used on dark backgrounds
- **Icon**: Square SVG for favicons and chat avatars (recommended: 512x512px)
- **Format**: PNG for logos, SVG for icons
- **Recommended height**: 100px for main logos

#### Step 3: Update Documentation URLs and Branding

Update all README files by finding and replacing the following patterns:

| Original | Replace with (for example) | Files Affected |
|----------|--------------|----------------|
| `https://cloud.sambanova.ai` | `https://cloud.custx.ai` | All READMEs, Streamlit apps |
| `https://community.sambanova.ai` | `https://community.custx.ai` | All READMEs |
| `https://sambanova.ai/` | `https://custx.ai/` | All READMEs |
| `SambaNova` | `Custx` | All READMEs (carefully!) |
| `github.com/sambanova/ai-starter-kit` | `github.com/custx/ai-starter-kit` | All READMEs |

**Important notes**:

- Be careful when replacing "SambaNova" - some references may need to remain (e.g., in technical attributions)
- Preserve license and attribution requirements
- Test all links after replacement

#### Step 4: Update the main README file

- In the main [README file](./README.md), replace all instructions related to SambaCloud with your custom platform (e.g., CustxCloud)
- Remove all specific instructions related to SambaManaged/SambaStack

## Section 3: AI Starter Kits UI Customization

### 3.1 Branding Customization

> See Section 5 Recommended White Labeling Tools and Automation to see bulk edition suggestions.

#### 3.1.1 Logo Replacement

**Logo usage in code**:
No code changes needed if you keep the same filenames as suggested in Section 2: Documentation Customization.

#### 3.1.2 Color Scheme Updates

Each Streamlit app has a theme configuration file:

**Location**: `[kit_name]/.streamlit/config.toml`

**Example**: `enterprise_knowledge_retriever/.streamlit/config.toml`

```toml
[theme]
base = "light"
primaryColor = "#250E36"              # SambaNova purple - CHANGE THIS
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F2F2F2"
textColor = "#000000"
font = "sans serif"
```

**Your brand colors**:

- Primary Color: `______` (used for buttons, links, highlights)
- Background: `______`
- Secondary Background: `______`
- Text Color: `______`

Update all kit config files with your desired color scheme.

> Additionally, there are some extra style configurations inside custom HTML in Streamlit apps to modify fonts, button hover colors, etc. Search for `<style>` labels inside `[kit_name]/.streamlit/app.py` and replace them if desired.

#### 3.1.3 Platform URL Updates in Code

**API key links in Streamlit apps**

All Streamlit apps contain this code:

```python
st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')
```

Replace it with your API portal link, for example: 
```python
st.markdown('Get your Custx API key [here](https://custx.ai/apis)')
```

#### 3.1.4 App Description Updates

Optionally, you can customize the app description messages (the default description shown on the screen when a kit is launched).

To change the kit description, edit the `[kit]/strwamlit/app_description.yaml` with your desired description to display.

### 3.2 Configuration for API Keys and Base URLs

#### 3.2.1 Understanding Current Configuration

The AI Starter Kits repository uses environment variables and config.yaml files for configuration:

**Primary environment file**: `.env` (see [.env-example](./env-exmple))

```bash
# Your custom cloud API key
SAMBANOVA_API_KEY=your-api-key-here

# Other utils variables
...
```

**Application config file**: `[kit]/config.yaml`

**How it works**:
1. Applications read from `.env` file or environment variables
2. Streamlit apps use `utils/visual/env_utils.py` for credential management
3. Backend code uses `langchain-sambanova` package which is injected with SAMBANOVA_API_KEY and SAMBANOVA_API_BASE variables
4. Streamlit apps use `[kit]/config.yaml` for setting the behavior specifics of the kit, including default models to use and if production mode is enabled

> production mode controls whether the API_KEY is taken from the .env file or prompted from the user when a new session starts.

#### 3.2.2 Required Code Changes for Base URL Input

Each kit requires the base API URL and API keys. These are requested from the user when the Streamlit app is launched (the API key is taken from the `.env` file if production mode is not enabled).
The base API URL is prefilled with the SambaCloud API URL, and the input fields contain SambaNova callouts by default; for example `Insert your SAMBANOVA_API_KEY`.

To change the input field text and prefilled values, rename the variables and update the values passed under the hood to the SambaNova LangChain wrappers. To do this:

1. In `./utils/visual/env_utils.py`, find and replace all occurrences of:
    - `SAMBANOVA_API_BASE` to `<CUSTX>_API_BASE`
    - `SAMBANOVA_API_KEY` to `<CUSTX>_API_KEY`

2. In each kit's Streamlit app `[kit_name]/streamlit/app.py`, find and replace all occurences of
    - `SAMBANOVA_API_BASE` to `<CUSTX>_API_BASE`
    - `SAMBANOVA_API_KEY` to `<CUSTX>_API_KEY`

3. Change the default prefilled base API URL in all kit Streamlit apps (`[kit_name/streamlit/app.py]`):
    - Update the `<CUSTX>_API_BASE` value in the `additional_env_vars` dictionary for the URL of your managed API:

    ``` python
    additional_env_vars = {'<CUSTX>_API_BASE': 'https://api.sambanova.ai/v1'}
    ```

    to:

    ``` python
    additional_env_vars = {'<CUSTX>_API_BASE': 'https://api.custx.ai/v1'}
    ```

4. Rename the base env variable in the [.env](./.env-sample) file:
    - Replace - `SAMBANOVA_API_KEY` to `<CUSTX>_API_KEY`

#### 3.2.3 Model List Customization

Each Streamlit app has set some default models to use (LLMs, LVLMs, and embeddings) that can be customized.

**Step 1**: Update `[kit]/config.yaml` with default models to use. For example:

```yaml
# enterprise_knowledge_retriever/config.yaml
llm:
  "model": "gpt-oss-120b"
  "temperature": 0.0
  "max_tokens": 8192

embedding_model:
  "model": "E5-Mistral-7B-Instruct"

# ... rest of config
```

**Step 2**: Update models selectors (when available)

Some kits includes a model selector that allow users to try different models in the same app. Those kits are:

- Enterprise Knowledge Retriever
- Multimodal Knowledge Retriever
- Search Assistant 

When a user selects one of these models, it is used instead of the default model set in the `[kit]/config.yaml` configuration file. To modify the list of models displayed in the model selector field, update: the `LLM_MODELS` constant inside `[kit]/app.py`.

**Example** from `enterprise_knowledge_retriever/streamlit/app.py` (lines 33-44):

```python
LLM_MODELS = [
    'gpt-oss-120b',
    'Llama-4-Maverick-17B-128E-Instruct',
    'Meta-Llama-3.3-70B-Instruct',
    'DeepSeek-R1-Distill-Llama-70B',
    'DeepSeek-R1',
    'DeepSeek-V3-0324',
    'DeepSeek-V3.1',
    'DeepSeek-V3.1-Terminus',
    'Meta-Llama-3.1-8B-Instruct',
    'Qwen-32B',
    'Qwen-235B',
]
```

To check what models are available in your platform, run the following command:

```bash
    curl --request GET \
    --url https://api.custx.ai/v1/models
```

### 3.4 Verification and Testing

#### 3.4.1 Testing Procedures

Verify each Streamlit app. For example:

```bash
# Test Enterprise Knowledge Retriever
cd enterprise_knowledge_retriever
streamlit run streamlit/app.py
```

<details>
    <summary>Individual App Testing Guide</summary>

For each kit, test the following:

| Kit | Test Case | Expected Result |
|-----|-----------|-----------------|
| **Enterprise Knowledge Retriever** | Upload a PDF, ask a question | Should retrieve relevant context and answer |
| **Multimodal Knowledge Retriever** | Upload a document with images | Should process both text and images |
| **Function Calling** | Trigger a function (e.g., get_time) | Should execute function and return result |
| **Search Assistant** | Ask a question requiring web search | Should search and synthesize answer |
| **Benchmarking** | Run a benchmark on a model | Should complete and show results |
| **Financial Assistant** | Query stock data or financial info | Should retrieve and display data |
| **Document Comparison** | Upload two documents to compare | Should highlight differences |
| **Data Extraction** | Extract data from a document | Should extract structured data |
| **Eval Jumpstart** | Run an evaluation | Should execute and show metrics |

- Verify logo appears in sidebar
- Verify page icon in browser tab
- Check that base URL input field is visible
- Test a simple query
- Verify chat avatar uses your icon
- Check that API key link points to your cloud URL
</details>

#### 3.4.2 Checklist

**Branding checklist**:

- [ ] All logos replaced (dark, light, icon)
- [ ] All README files updated with your branding
- [ ] All URLs updated (platform, community, website)
- [ ] Streamlit theme colors updated 
- [ ] API key links updated in all Streamlit apps
- [ ] License and attribution preserved

**Configuration checklist**:

- [ ] `.env` file created and configured
- [ ] `SAMBANOVA_API_BASE`, and `SAMBANOVA_API_KEY` variables set to your API variables
- [ ] API key valid and tested
- [ ] Model lists reviewed and customized

**Testing checklist**:

- [ ] Streamlit apps launch successfully
- [ ] API connection works with custom base URL
- [ ] Models load and respond correctly
- [ ] Logos display correctly in UI
- [ ] No broken links in documentation
- [ ] No references to old branding remain

> Make sure the name of your virtual environment is the same as the Makefile variable name `VENV_PATH`

## Section 4: Notebooks and Quickstarts Customization:

Each AI Starter Kit includes a Jupyter notebook demonstrating how to use the kit’s capabilities. By default, these notebooks reference SambaCloud's endpoints and environment variable names (API base URL and API key names). Use your cloud's URL and API env variable names so they point to your platform instead of `cloud.sambanova.ai` or `api.sambanova.ai`.

**Note**: By default, the backend uses `langchain-sambanova` wrappers, which we recommend. Alternatively, you can use `langchain-openai` wrappers with your API key and base URL (our endpoints are OpenAI-compatible), or implement your own LangChain wrappers if needed.

### 4.1 What to Change

- **Base URL strings**: Replace any hardcoded `https://api.sambanova.ai/v1` or `https://cloud.sambanova.ai` with your cloud URL (e.g., `https://api.custx.ai/v1`).
- **Env var names**: Replace `SAMBANOVA_API_KEY` (and `SAMBANOVA_API_BASE`, if present) with your chosen env var names (e.g., `CUSTX_API_KEY`, `CUSTX_API_BASE`).
- **Headers/config dicts**: Some cells build headers or client configs (e.g., `{"Authorization": f"Bearer {os.environ['SAMBANOVA_API_KEY']}"}`). Point them to your env vars after renaming.

### 4.2 How to Update the Notebooks

1. Identify notebooks
   - Quickstarts: `quickstart/*.ipynb` 
   - Kit notebooks: `[kit]/notebooks/*.ipynb` (e.g., `function_calling/notebooks`, `multimodal_knowledge_retriever/notebooks`)

2. Search and replace (per notebook)
   - Look for `cloud.sambanova.ai`, `api.sambanova.ai`, `SAMBANOVA_API_KEY`, `SAMBANOVA_API_BASE`.
   - Replace with your URLs and env var names (e.g., `cloud.custx.ai`, `<CUSTX>_API_KEY`, `<CUSTX>_API_BASE`).
   - If a cell sets `os.environ["SAMBANOVA_API_KEY"] = "..."`, rename the key and remove any sample key values.

### 4.3 Quick Checklist

- [ ] All base URLs in notebooks point to your domain
- [ ] All env var names use your chosen keys (not `SAMBANOVA_API_KEY`)
- [ ] Auth headers/config blocks reference the new env vars
- [ ] No sample keys or old domains remain in markdown or code cells

## Section 5: Recommended White Labeling Tools and Automation

To efficiently customize documentation across multiple files, we recommend the following tools:

#### Option 1: IDE Find and Replace

You can replace in bulk all links and constants using the VS Code Find and Replace tool:

- Open VS Code and open the AI Starter Kit workspace
- Open the search default panel in the extensions bar
- Add the link or constant to change in the Search field, and ensure `Match Case` option is enabled
- Add the desired new link or constant in the Replace field
- Add the desired file types to search in files to include, eg.(`*.md, *streamlit/app.py`)

#### Option 2: Code Assistants Editing

The code assistant's file editing feature allows you to apply AI-assisted changes across multiple files simultaneously.

**Setup**:

1. Install a code assitant-powered IDE (Cursor, Windsurf) or VS Code with any code copilot extension (Github Copilot, Cline, Codex, Roocode, etc...)
2. Open the AI Starter Kits repository in VS Code or your selected IDE

**Usage**:

```txt
# Example prompts for LLm assitant edits:

1. "Replace all occurrences of 'https://cloud.sambanova.ai' with
   'https://cloud.custx.ai' in all README.md files"

2. "In all Streamlit app.py files, replace the hardcoded URL
   'https://cloud.sambanova.ai/api' with 'https://cloud.custx.ai/api'"

3. "In all Streamlit app.py files, replace the uppercase 'SAMBANOVA_API_KEY' for 'CUSTX_API_KEY' and 'SAMBANOVA_API_BASE' for 'CUSTX_API_BASE' "
```

#### Option 3: Command Line Tools

For developers comfortable with command-line tools:

**Using `sed` (macOS/Linux)**:

```bash
# Replace cloud URL in all README files
find . -name "README.md" -type f -exec sed -i '' \
  's|https://cloud.sambanova.ai|https://cloud.custx.ai|g' {} +

# Replace company name in all README files (be careful!)
find . -name "README.md" -type f -exec sed -i '' \
  's/SambaNova/YourCompany/g' {} +

# Update community links
find . -name "README.md" -type f -exec sed -i '' \
  's|https://community.sambanova.ai|https://community.custx.ai|g' {} +
```

**Using `ripgrep` + `sd` (modern alternative)**:

```bash
# Install sd: brew install sd

# Replace URLs
rg "cloud.sambanova.ai" -l | xargs sd "cloud.sambanova.ai" "cloud.custx.ai"
```

## Section 6: Merging Changes to Your Main

### 6.1 Formatting & Linting

This guide explains how to run formatting, linting, and type-checking manually using command-line tools, or automatically using the provided Makefile and GitHub Workflows.

#### 6.1.1 Manual Usage (Direct Commands)

Use these commands if you prefer running tools manually.

**1. Install dependencies**
With your virtual env activated, ensure all dependencies are installed

``` bash
  pip install uv 
  uv pip install base-requirements.txt
```

**2. Format all code**

```bash
ruff format .
```

**3. Sort imports**

```bash
ruff check --select I --fix .
```

**4. Lint the code and automatically fix issues**

```bash
ruff check --fix .
```

**5. Run type checks with mypy**

```bash
mypy --explicit-package-bases .
```

Running these ensures:

- Consistent formatting
- Correct import ordering
- Linted and auto-fixed code
- Type-safe modules and early bug detection

#### 6.1.2 Using the Makefile (Recommended)

The [Makefile](makefile) provides shortcuts for the most common tasks.

**Format + sort imports + auto-fix + lint**

```bash
make format
```

This runs under the hood:

```bash
ruff format .
ruff check --select I --fix .
ruff check --fix .
```

**Lint + type-check**

```bash
make lint
```

This runs under the hood:

```bash
ruff check --select I --fix .
ruff check --fix .
mypy --explicit-package-bases .
```

**Format + sort imports + auto-fix + lint + type-check**

```bash
make format-lint
```

This runs under the hood:

```bash
ruff format .
ruff check --select I --fix .
ruff check --fix .
mypy --explicit-package-bases .
```

#### 6.1.3 GitHub Workflows (CI)

Every Pull Request (PR) automatically triggers checks that:

- Format your code (`ruff format`)
- Sort and validate imports
- Run ruff lint checks
- Run mypy --explicit-package-bases .
- Fail the PR if any issues are found

#### Summary
| Task               | Manual Command                     | Makefile Target   |
|--------------------|------------------------------------|--------------------|
| Format code        | `ruff format .`                    | `make format`      |
| Sort imports       | `ruff check --select I --fix .`    | `make format`      |
| Lint & auto-fix    | `ruff check --fix .`               | `make format`      |
| Type checking      | `mypy --explicit-package-bases .`  | `make lint`        |

GitHub Workflows automatically run all of these checks on every PR.

### 6.2 Testing

#### 6.2.1 Manual Usage

**Quick per-kit checks** (run from repo root; requires `.env` with API keys):

Run each kit’s tests:

   ```bash
   python [kit_name]/tests/*.py                         
   ```

#### 6.2.2 Using the Makefile

The Makefile provides a dedicated virtual environment for running the full test suite:
- `make setup-test-suite` — create the dedicated test venv and install test deps.
- `make clean-test-suite` — remove the test venv and unset the local pyenv pin.
- Pair these with `./run_tests.sh ...` to execute the suites.

#### 6.2.3 GitHub Workflows (CI)

- Pull Requests trigger the same core Python checks plus the test harness (equivalent to `./run_tests.sh`), depending on available secrets.
- Ensure repo secrets such as `<CUSTX>_API_KEY` and `SERPAPI_API_KEY` are set. Tests that rely on external APIs will be skipped or fail early if these secrets are missing.

#### Summary
| Task                            | Manual Command / Script                        | Makefile Target      |
|---------------------------------|-----------------------------------------------|----------------------|
| Set up test env                 | `make setup-test-suite`                       | `setup-test-suite`   |
| Run full local test suite       | `./run_tests.sh local --skip-streamlit`       | (use script)         |
| Run Docker test suite           | `./run_tests.sh docker --skip-streamlit`      | (use script)         |
| Run all (local + Docker)        | `./run_tests.sh all --skip-streamlit`         | (use script)         |
| Per-kit quick check             | `python tests/<kit>_test.py`                  | n/a                  |
| Clean test env                  | `make clean-test-suite`                       | `clean-test-suite`   |

### 6.3 Merging Changes

After running all required checks locally, open a Pull Request to the main branch. The GitHub workflows will run Python checks and the full test suite. You may safely merge into main once all checks and tests pass.

These workflows require the current secrets to be set in your repository:

- `<CUSTX>_API_KEY`
- `SERPAPI_API_KEY`

Ensure these are set before sending your PR.

## Section 7: Deployment Overview

The AI Starter Kits repository ships with utilities to help you move from customization to production, but the exact topology (Kubernetes, VMs, on-prem) is customer-specific. Use the building blocks below to assemble a deployment that matches your standards/environments.

### 7.1 What This Repo Provides for Deployment

- **Containerization**: `Dockerfile` builds one reusable image that contains all kits. You choose which kit to start when you run the container. `docker-startup.sh` loads `.env`, applies prod/test tweaks, and then runs the command you pass.
- **Environment bootstrap**: `Makefile` installs Python, dependencies, parsing service, and system tools (Poppler/Tesseract/libheif) for local or image builds; `make docker-build` produces the test image used by CI scripts.
- **Configuration hardening**: Each kit `config.yaml` supports `prod_mode`; `utils/prod/update_config.py` can batch-set `prod_mode`, disable risky tools (e.g., Python REPL), and set Streamlit ports.
- **Secrets and env management**: `.env-example` defines required variables (`SAMBANOVA_API_KEY`, optional `SAMBANOVA_API_BASE`, third-party keys). Streamlit apps pull from env at startup.
- **Smoke/regression checks**: `run_tests.sh` and `tests/test_framework.py` run CLI and Streamlit availability checks locally or in Docker; useful before promoting artifacts.

### 7.2 Generic Production Deployment Flow (Adapt to Your Stack)

1. **Prepare config**
   - Copy `.env-example` to `.env` and set API keys, base URLs, and any third-party keys.
   - For the kits you’ll host, set `prod_mode: true` and desired default models in `[kit]/config.yaml` (or run `python utils/prod/update_config.py --mode prod --port <port>`).
   - Lock Streamlit port via `STREAMLIT_PORT := <port>` in `Makefile` if you need a fixed value behind a proxy.

2. **Build the image**

   ```bash
   docker build -t <registry>/<image>:<tag> -f Dockerfile .
   ```

   - If you need the parsing service, enable `PARSING=true` at build/run time and ensure its port is exposed internally.

3. **Pick a kit to run (per container)**
   - The image includes all kits; you choose which one starts.
   - Simple option (auto-wires ports and `.env`):  

     ```bash
     make docker-run-kit KIT=enterprise_knowledge_retriever
     ```

   - Manual option (replace the kit name as needed):

     ```bash
     docker run --rm -p 8501:8501 --env-file .env <registry>/<image>:<tag> \
       /bin/bash -c "cd enterprise_knowledge_retriever && \
       streamlit run streamlit/app.py --server.port 8501 --server.address 0.0.0.0"
     ```
  
   - If the container exits immediately, check that you passed a command; the default `CMD ["make","run"]` is only a placeholder.

4. **Run and verify**

   ```bash
   docker run --rm -p 8501:8501 \
     --env-file .env \
     <registry>/<image>:<tag>
   ```
  
   - Smoke test with `./run_tests.sh local --skip-streamlit` (or include Streamlit if you allow browsers in your env). For air-gapped CI, mount `test_results/` as needed.

5. **Harden and promote**
   - Front with your standard reverse proxy (TLS termination, auth, rate limiting). Common choices: Nginx/Envoy/ALB/Ingress.
   - Inject secrets at runtime via your secret manager (KMS/SM/HashiCorp Vault) instead of baking them into images.
   - Add health checks hitting `/:` or a lightweight ping endpoint; container should be configured with `HEALTHCHECK` if your platform uses it.
   - For Kubernetes, wrap the container in a Deployment/StatefulSet, mount config as ConfigMap/Secret, and expose via Service/Ingress.

### 7.3 Production Considerations (Apply per Environment)

- **Data handling**: Mount temp storage for uploads if your kit processes documents; add lifecycle rules to clean uploads. Validate file size/type at the proxy and app layer.
- **Network egress**: Whitelist outbound access only to your model endpoints and required third parties (e.g., search APIs) if using kits that call the web.
- **Security**: Enforce TLS, origin restrictions, and authentication (OIDC/SAML header injection via proxy is common). Disable Python REPL/tooling via `prod_mode` where not needed.
- **Updates**: Rebuild images when upstream kits change; use `make docker-build` in CI, then deploy with your standard promotion pipeline.

## Additional Documentation and Resources

<details>
    <summary>Keeping Up with SambaNova Updates</summary>

As SambaNova continues to improve the AI Starter Kit, you'll want to incorporate updates while maintaining your customizations.

#### Maintaining a Fork with Selective Merging

```bash
# Add SambaNova repo as upstream remote
git remote add upstream https://github.com/sambanova/ai-starter-kit.git

# Fetch updates from SambaNova
git fetch upstream

# Review changes
git log main..upstream/main --oneline

# Merge specific commits or files
git cherry-pick <commit-hash>

# Or merge entire branch and resolve conflicts
git merge upstream/main
```

#### Monitoring for Updates

**GitHub watch**

- Click "Watch" on the SambaNova repository
- Select "Custom" → "Releases"

### Support Information

#### Internal Support Setup

1. **Documentation**
   - Maintain a custom README_CUSTOM.md with your specific setup notes
   - Document any additional customizations
   - Keep a changelog of modifications

2. **Issue tracking**
   - Use GitHub Issues in your forked repository
   - Tag issues: `branding`, `configuration`, `upstream-merge`, `bug`, `enhancement`

3. **Knowledge base**
   - Document common issues and solutions
   - Create troubleshooting guides for your team
   - Maintain a FAQ

#### When to Contact SambaNova

Contact SambaNova support for:

- Issues with core functionality (not related to your customizations)
- Questions about model availability or performance
- API-related problems
- Security vulnerabilities

**Do not contact SambaNova for**:

- Issues introduced by your customizations
- Branding-related questions
- Custom code you've added

### Support and Issue Tracking

#### For Issues with SambaNova's Original Code

Report issues to SambaNova:
- GitHub Issues: `https://github.com/sambanova/ai-starter-kit/issues`
- Community Forum: `https://community.sambanova.ai`

#### For Your Customized Version

Set up your own support channels (suggested):

1. Create an internal issue tracker
2. Set up a dedicated Slack/Teams channel
3. Document known issues in your fork's README
4. Maintain a CHANGELOG.md for your customizations

---

</details>

<details>
    <summary>Official SambaNova Resources</summary>

**Documentation**:

- SambaNova Docs: https://docs.sambanova.ai
- API Reference: https://docs.sambanova.ai/cloud/api-reference
- Integration Guides: https://docs.sambanova.ai/cloud/docs/integrations

**Community**:

- Community Forum: https://community.sambanova.ai
- GitHub Issues: https://github.com/sambanova/ai-starter-kit/issues
- GitHub Discussions: https://github.com/sambanova/ai-starter-kit/discussions

**Code repositories**:

- AI Starter Kit: https://github.com/sambanova/ai-starter-kit
- Agents: https://github.com/sambanova/agents
- Integrations: https://github.com/sambanova/integrations

</details>

<details>
    <summary>Third-Party Tools & Resources</summary>

**Development tools**:

- VS Code: https://code.visualstudio.com
- GitHub Copilot: https://github.com/features/copilot
- Streamlit: https://docs.streamlit.io

**Documentation tools**:

- Markdown Guide: https://www.markdownguide.org
- Pandoc: https://pandoc.org
- MkDocs: https://www.mkdocs.org

**CI/CD & automation**:

- GitHub Actions: https://docs.github.com/actions
- Pre-commit hooks: https://pre-commit.com

**Monitoring & debugging**:

- LangSmith: https://docs.smith.langchain.com
- MLflow: https://mlflow.org

</details>

<details>
    <summary>Best Practices</summary>

**Version Control**:
1. Always work in feature branches
2. Use descriptive commit messages
3. Tag releases (e.g., `v1.0.0-custx`)
4. Document changes in CHANGELOG.md

**Configuration Management**:
1. Never commit `.env` files
2. Use `.env.example` as template
3. Document all configuration options
4. Use environment-specific configs (dev, staging, prod)

**Testing**:
1. Test each component individually
2. Perform end-to-end integration tests
3. Validate all external links
4. Check UI responsiveness
5. Test with different API keys and endpoints

**Documentation**:
1. Keep README files up to date
2. Document all customizations
3. Maintain a decision log
4. Include troubleshooting guides

**Security**:
1. Rotate API keys regularly
2. Use environment variables for secrets
3. Enable authentication in production
4. Review code for security vulnerabilities
5. Keep dependencies updated

</details>

<details>
    <summary>Troubleshooting Common Issues</summary>

#### Issue: "API Key Invalid" Error

**Symptoms**: Applications fail to authenticate

**Solutions**:
1. Verify API key is correctly set in `.env`
2. Check for extra spaces or newlines in key
3. Ensure key hasn't expired
4. Verify key has correct permissions

#### Issue: "Module Not Found" Error

**Symptoms**: Import errors when running apps

**Solutions**:
1. Ensure you're in correct directory
2. Activate virtual environment
3. Install requirements:

```bash
pip install uv
uv pip install -r base-requirements.txt
uv pip install -r [kit]/requirements.txt
```

4. Check Python version (3.10+ required)

#### Issue: Streamlit App Won't Start

**Symptoms**: Port already in use, or app crashes on startup

**Solutions**:

1. Check if another instance is running:

```bash
lsof -i :8501  # Default Streamlit port
kill -9 <PID>  # Kill the process
```

2. Use a different port:

```bash
streamlit run app.py --server.port 8502
```

3. Check logs for specific errors
4. Verify all dependencies installed

#### Issue: Logo Not Displaying

**Symptoms**: Broken image icon in UI or README

**Solutions**:

1. Check file paths are correct
2. Verify image files exist and are readable
3. Check file permissions
4. Verify file format (PNG, SVG)
5. Clear browser cache

#### Issue: Model Not Available

**Symptoms**: "Model not found" or similar error

**Solutions**:

1. Check model name spelling (case-sensitive)
2. Verify model is deployed on your cloud
3. Check API endpoint supports the model:

```bash
curl --request GET \
  --url https://api.sambanova.ai/v1/models
```

4. Try a different model
5. Check model list configuration

</details>

## Document Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-12 | Initial release |

---

## License & Attribution

This guide is provided for SambaManaged and SambaStack customers. The original SambaNova AI Starter Kits and Agents application are licensed under their respective licenses. Ensure you comply with all license requirements when customizing and deploying these applications.

AI Starter Kits: Copyright © SambaNova Systems.
