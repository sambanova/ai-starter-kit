# White Labeling and Customization Guide

## SambaNova AI Starter Kit & Agents Application

**Target Audience**: SambaManaged customers with white-labeled cloud infrastructure (e.g., `cloud.custx.ai`)

**Version**: 1.0
**Last Updated**: 2025-11-13

---

## Table of Contents

1. [Overview](#overview)
2. [Section 1: Documentation Customization](#section-1-documentation-customization)
3. [Section 2: AI Starter Kit White Labeling](#section-2-ai-starter-kit-white-labeling)
4. [Section 3: Agents Application Configuration](#section-3-agents-application-configuration)
5. [Appendix A: Required Code Changes](#appendix-a-required-code-changes)
6. [Appendix B: Support & Resources](#appendix-b-support--resources)

---

## Overview

This guide provides comprehensive instructions for customizing and white-labeling SambaNova's AI Starter Kit and Agents Application for your deployment. As a SambaManaged customer with your own branded cloud infrastructure, you can adapt these resources to:

- Replace SambaNova branding with your company's branding
- Configure applications to use your custom cloud URL (e.g., `cloud.custx.ai`)
- Customize documentation and user interfaces
- Add custom LLM providers to the Agents application

### What This Guide Covers

- **Documentation**: How to update READMEs, links, and documentation content
- **AI Starter Kit**: Branding customization, configuration for custom cloud endpoints, model selection
- **Agents App**: Enabling admin panel and adding custom LLM providers
- **Tools & Automation**: Recommended tools for efficient bulk updates

### Prerequisites

- Git installed and configured
- Python 3.10+ installed
- Access to your white-labeled cloud infrastructure
- API keys for your cloud platform
- Basic familiarity with:
  - Git and GitHub workflows
  - Environment variables and configuration files
  - Text editors or IDEs
  - Command line interface

---

## Section 1: Documentation Customization

### 1.1 Documentation Sources and Structure

The AI Starter Kit documentation is organized as follows:

#### **Main Documentation Files**

```
ai-starter-kit/
├── README.md                                    # Root README
├── CONTRIBUTING.md                              # Contribution guidelines
├── images/                                      # Logo and icon assets
│   ├── SambaNova-dark-logo-1.png               # Main dark theme logo
│   ├── SambaNova-light-logo-1.png              # Main light theme logo
│   └── SambaNova-icon.svg                       # Icon/favicon
└── [kit_name]/
    └── README.md                                # Kit-specific README
```

#### **Kit-Specific READMEs**

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
- `e2e_fine_tuning/README.md`

### 1.2 Documentation Tooling

**Format**: Standard Markdown (.md files)
**No specialized generators**: The documentation uses GitHub-flavored Markdown without external documentation generators like MkDocs or Sphinx.

This means customization is straightforward:
- Edit files directly with any text editor
- Preview in GitHub or using Markdown preview tools
- No build process required for documentation

### 1.3 Customization Process

#### **Step 1: Fork or Clone the Repository**

```bash
# Clone the repository
git clone https://github.com/sambanova/ai-starter-kit.git
cd ai-starter-kit

# Create a new branch for your customizations
git checkout -b whitelabel-customization
```

#### **Step 2: Identify Branding Elements to Replace**

All READMEs contain the following SambaNova-specific elements:

1. **Logo** (at the top of each README):
```markdown
<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SambaNova-light-logo-1.png" height="100">
  <img alt="SambaNova logo" src="./images/SambaNova-dark-logo-1.png" height="100">
</picture>
</a>
```

2. **Cloud Platform URL**: `https://cloud.sambanova.ai`
3. **Community URL**: `https://community.sambanova.ai`
4. **Company Website**: `https://sambanova.ai/`
5. **GitHub Repository**: `https://github.com/sambanova/ai-starter-kit`

#### **Step 3: Replace Logo Files**

Place your company's logo files in the `images/` directory:

```bash
# Backup original logos (optional)
mv images/SambaNova-dark-logo-1.png images/SambaNova-dark-logo-1.png.backup
mv images/SambaNova-light-logo-1.png images/SambaNova-light-logo-1.png.backup
mv images/SambaNova-icon.svg images/SambaNova-icon.svg.backup

# Add your logo files (keep the same filenames for easier migration)
cp /path/to/your-dark-logo.png images/SambaNova-dark-logo-1.png
cp /path/to/your-light-logo.png images/SambaNova-light-logo-1.png
cp /path/to/your-icon.svg images/SambaNova-icon.svg
```

**Logo Specifications**:
- **Dark Logo**: Used on light backgrounds
- **Light Logo**: Used on dark backgrounds
- **Icon**: Square SVG for favicons and chat avatars (recommended: 512x512px)
- **Format**: PNG for logos, SVG for icons
- **Recommended height**: 100px for main logos

#### **Step 4: Update Documentation URLs and Branding**

You'll need to update all README files. See section 1.4 for recommended automation tools.

**Find and replace these patterns:**

| Original | Replace With | Files Affected |
|----------|--------------|----------------|
| `https://cloud.sambanova.ai` | `https://cloud.custx.ai` | All READMEs, Streamlit apps |
| `https://community.sambanova.ai` | `https://community.custx.ai` | All READMEs |
| `https://sambanova.ai/` | `https://custx.ai/` | All READMEs |
| `SambaNova` | `YourCompany` | All READMEs (carefully!) |
| `github.com/sambanova/ai-starter-kit` | `github.com/custx/ai-starter-kit` | All READMEs |

**Important Notes**:
- Be careful when replacing "SambaNova" - some references may need to remain (e.g., in technical attributions)
- Preserve license and attribution requirements
- Test all links after replacement

### 1.4 Recommended AI Tools and Automation

To efficiently customize documentation across multiple files, we recommend the following tools:

#### **Option 1: GitHub Copilot Multi-File Editing** (Recommended for VS Code Users)

GitHub Copilot's multi-file editing feature allows you to apply AI-assisted changes across multiple files simultaneously.

**Setup**:
1. Install VS Code with GitHub Copilot extension
2. Open the AI Starter Kit repository in VS Code
3. Access Copilot Edits: `View` → `Command Palette` → `GitHub Copilot: Open Copilot Edits`

**Usage**:
```
# Example prompts for Copilot Edits:

1. "Replace all occurrences of 'https://cloud.sambanova.ai' with
   'https://cloud.custx.ai' in all README.md files"

2. "Update the logo links in all README files to use 'YourCompany'
   instead of 'SambaNova'"

3. "In all Streamlit app.py files, replace the hardcoded URL
   'https://cloud.sambanova.ai/apis' with 'https://cloud.custx.ai/apis'"
```

**Add Custom Instructions**:

Create `.github/copilot-instructions.md`:
```markdown
When updating documentation:
- Preserve technical accuracy
- Maintain markdown formatting
- Keep code examples intact
- Only update branding and URLs
```

#### **Option 2: Command Line Tools**

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

#### **Option 3: Pandoc for Advanced Transformations**

For complex documentation transformations or format conversions:

```bash
# Install Pandoc
brew install pandoc  # macOS
# or: sudo apt-get install pandoc  # Linux

# Example: Convert and transform markdown
pandoc README.md -o README_transformed.md --lua-filter=custom-filter.lua
```

Create a Lua filter (`custom-filter.lua`) for advanced replacements:
```lua
function Link(el)
  el.target = el.target:gsub("cloud.sambanova.ai", "cloud.custx.ai")
  return el
end
```

#### **Option 4: Documentation Linting and Validation**

Ensure consistency after changes:

```bash
# Install markdownlint
npm install -g markdownlint-cli

# Check all markdown files
markdownlint '**/*.md' --ignore node_modules

# Install markdown-link-check
npm install -g markdown-link-check

# Validate all links
find . -name "*.md" -exec markdown-link-check {} \;
```

#### **Option 5: CI/CD Automation with GitHub Actions**

Automate documentation validation in your forked repository:

Create `.github/workflows/docs-validation.yml`:
```yaml
name: Documentation Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Lint Markdown
        uses: nosborn/github-action-markdown-cli@v3.3.0
        with:
          files: .

      - name: Check Links
        uses: gaurav-nelson/github-action-markdown-link-check@v1
        with:
          use-quiet-mode: 'yes'

      - name: Validate Branding
        run: |
          # Check that old URLs are not present
          if grep -r "cloud.sambanova.ai" .; then
            echo "Error: Found references to cloud.sambanova.ai"
            exit 1
          fi
```

### 1.5 Keeping Up with SambaNova Updates

As SambaNova continues to improve the AI Starter Kit, you'll want to incorporate updates while maintaining your customizations.

#### **Strategy 1: Maintaining a Fork with Selective Merging**

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

#### **Strategy 2: Automated Rebranding Script**

Create a rebranding script (`scripts/rebrand.sh`) to re-apply customizations after merging:

```bash
#!/bin/bash
# rebrand.sh - Reapply white labeling after upstream merge

echo "Reapplying white labeling..."

# Replace URLs
find . -name "README.md" -type f -exec sed -i '' \
  's|https://cloud.sambanova.ai|https://cloud.custx.ai|g' {} +

find . -name "*.py" -type f -path "*/streamlit/*" -exec sed -i '' \
  's|cloud.sambanova.ai/apis|cloud.custx.ai/apis|g' {} +

# Replace company names (carefully)
find . -name "README.md" -type f -exec sed -i '' \
  's/Get your SambaNova API key/Get your API key/g' {} +

echo "Rebranding complete!"
```

**Usage after merging upstream**:
```bash
# Merge updates
git merge upstream/main

# Reapply customizations
./scripts/rebrand.sh

# Review changes
git diff

# Commit
git add .
git commit -m "Reapply white labeling after upstream merge"
```

#### **Strategy 3: Configuration-Driven Approach**

Create a configuration file for your branding (`config/branding.json`):

```json
{
  "company_name": "YourCompany",
  "cloud_url": "https://cloud.custx.ai",
  "community_url": "https://community.custx.ai",
  "website_url": "https://custx.ai",
  "github_repo": "https://github.com/custx/ai-starter-kit",
  "logo_dark": "./images/SambaNova-dark-logo-1.png",
  "logo_light": "./images/SambaNova-light-logo-1.png",
  "logo_icon": "./images/SambaNova-icon.svg",
  "primary_color": "#250E36"
}
```

Create a Python script to apply branding from config:

```python
# scripts/apply_branding.py
import json
import re
from pathlib import Path

def load_branding_config():
    with open('config/branding.json') as f:
        return json.load(f)

def apply_branding_to_file(file_path, config):
    content = file_path.read_text()

    # Replace URLs
    content = content.replace('https://cloud.sambanova.ai', config['cloud_url'])
    content = content.replace('https://community.sambanova.ai', config['community_url'])
    content = content.replace('https://sambanova.ai/', config['website_url'])

    file_path.write_text(content)

def main():
    config = load_branding_config()

    # Apply to all READMEs
    for readme in Path('.').rglob('README.md'):
        apply_branding_to_file(readme, config)
        print(f"Updated: {readme}")

if __name__ == '__main__':
    main()
```

### 1.6 Support and Issue Tracking

#### **For Issues with SambaNova's Original Code**

Report issues to SambaNova:
- GitHub Issues: `https://github.com/sambanova/ai-starter-kit/issues`
- Community Forum: `https://community.sambanova.ai`

#### **For Your Customized Version**

Set up your own support channels:
1. Create an internal issue tracker
2. Set up a dedicated Slack/Teams channel
3. Document known issues in your fork's README
4. Maintain a CHANGELOG.md for your customizations

---

## Section 2: AI Starter Kit White Labeling

### 2.1 Getting Started

#### **Installation and Setup**

1. **Fork the Repository**
```bash
# Fork via GitHub UI or:
gh repo fork sambanova/ai-starter-kit --clone
cd ai-starter-kit
```

2. **Install Dependencies**
```bash
# Install base requirements
pip install -r base-requirements.txt

# Install kit-specific requirements (example for Enterprise Knowledge Retriever)
cd enterprise_knowledge_retriever
pip install -r requirements.txt
```

3. **Set Up Environment Variables**
```bash
# Copy example environment file
cp .env-example .env

# Edit .env with your credentials
nano .env
```

### 2.2 Branding Customization

#### **2.2.1 Logo Replacement**

**Locations where logos are used:**

1. **README Files** (11 files)
   - Root README.md
   - Each kit's README.md

2. **Streamlit Applications** (9 apps)
   - Page icon (browser tab)
   - Sidebar logo
   - Chat avatars

**Logo Files to Replace**:

```
images/
├── SambaNova-dark-logo-1.png    # Used in READMEs and Streamlit sidebars
├── SambaNova-light-logo-1.png   # Used in READMEs (dark mode)
└── SambaNova-icon.svg           # Used in browser tabs and chat avatars
```

**Replacement Steps**:

```bash
# 1. Prepare your logo files with the same names
cp /path/to/your/dark-logo.png images/SambaNova-dark-logo-1.png
cp /path/to/your/light-logo.png images/SambaNova-light-logo-1.png
cp /path/to/your/icon.svg images/SambaNova-icon.svg

# 2. Verify file sizes (recommended for web performance)
du -h images/SambaNova-*.png images/SambaNova-*.svg

# 3. Test logo display in a Streamlit app
cd enterprise_knowledge_retriever
streamlit run streamlit/app.py
```

**Logo Usage in Code**:

```python
# Streamlit apps reference logos like this:
# enterprise_knowledge_retriever/streamlit/app.py (line 293)
logo_path = os.path.join(repo_dir, 'images', 'SambaNova-dark-logo-1.png')
st.image(logo_path, width=150)

# Chat avatar (line 509)
avatar = os.path.join(repo_dir, 'images', 'SambaNova-icon.svg')
with st.chat_message('assistant', avatar=avatar):
    st.markdown(response)
```

No code changes needed if you keep the same filenames!

#### **2.2.2 Color Scheme Updates**

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

**Update all 9 kit config files**:

```bash
# List all Streamlit config files
find . -name "config.toml" -path "*/.streamlit/*"

# Update primary color in all configs (example: change to blue)
find . -name "config.toml" -path "*/.streamlit/*" -exec sed -i '' \
  's/primaryColor = "#250E36"/primaryColor = "#0066CC"/g' {} +
```

**Your Brand Colors**:
- Primary Color: `______` (used for buttons, links, highlights)
- Background: `______`
- Secondary Background: `______`
- Text Color: `______`

#### **2.2.3 URL Updates in Code**

**Hardcoded URLs that need updating:**

1. **API Key Links in Streamlit Apps**

All Streamlit apps contain this markdown:
```python
st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')
```

**Files to update** (9 files):
- `enterprise_knowledge_retriever/streamlit/app.py` (line 308)
- `multimodal_knowledge_retriever/streamlit/app.py` (line 303)
- `function_calling/streamlit/app.py` (line 345)
- `search_assistant/streamlit/app.py` (line 278)
- `benchmarking/streamlit/streamlit_utils.py` (line 106)
- `document_comparison/streamlit/app.py` (line 251)
- `financial_assistant/streamlit/app.py`
- `eval_jumpstart/streamlit/app.py`
- `data_extraction/streamlit/app.py` (if exists)

**Bulk update command**:
```bash
# Replace API key URL in all Streamlit apps
find . -name "*.py" -path "*/streamlit/*" -exec sed -i '' \
  's|https://cloud.sambanova.ai/apis|https://cloud.custx.ai/apis|g' {} +
```

2. **README Documentation URLs**

See Section 1.3 for updating README files.

### 2.3 Configuration for Custom Cloud

#### **2.3.1 Understanding Current Configuration**

The AI Starter Kit uses environment variables for configuration:

**Primary Configuration File**: `.env`

```bash
# Your custom cloud configuration
SAMBANOVA_API_KEY=your-api-key-here
SAMBANOVA_API_BASE=https://cloud.custx.ai/v1  # Your custom cloud URL
```

**How it works**:
1. Applications read from `.env` file or environment variables
2. Streamlit apps use `utils/visual/env_utils.py` for credential management
3. Backend code uses `langchain-sambanova` package which respects `SAMBANOVA_API_BASE`

#### **2.3.2 Required Code Changes for Base URL Input**

**Current State**: Base URL input is available but not prominently displayed in all UIs.

**Location**: `utils/visual/env_utils.py` (lines 64-101)

The utility already supports `SAMBANOVA_API_BASE` as an optional parameter:

```python
def env_input_fields(additional_env_vars: Union[List[str], Dict[str, Any]] = None):
    # If SAMBANOVA_API_BASE in additional env vars, show it first
    if 'SAMBANOVA_API_BASE' in additional_env_vars:
        additional_vars['SAMBANOVA_API_BASE'] = st.text_input(
            'SAMBANOVA API BASE',
            value=st.session_state.get('SAMBANOVA_API_BASE', ''),
            type='password'
        )
```

**What needs to change**: Each Streamlit app needs to include `SAMBANOVA_API_BASE` in its `additional_env_vars`.

**Example for Enterprise Knowledge Retriever**:

Current code in `enterprise_knowledge_retriever/streamlit/app.py`:
```python
# Around line 366
initialize_env_variables(config.get('prod_mode', False))
```

**Change to**:
```python
# Add SAMBANOVA_API_BASE to the configuration
initialize_env_variables(
    config.get('prod_mode', False),
    additional_env_vars=['SAMBANOVA_API_BASE']  # ADD THIS
)
```

And update the credentials section:
```python
# Around line 306-315
with st.expander('Credentials'):
    # Current code
    api_key, additional_vars = env_input_fields()

    # Change to
    api_key, additional_vars = env_input_fields(['SAMBANOVA_API_BASE'])
```

**Apply this change to all 9 Streamlit apps**. See Appendix A for the complete list.

#### **2.3.3 Model List Customization**

**Current State**: Model lists are hardcoded in each Streamlit app.

**Example from `enterprise_knowledge_retriever/streamlit/app.py` (lines 33-44)**:
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
]
```

**Recommended Change**: Move model lists to `config.yaml` for easier customization.

**Step 1**: Update `config.yaml` to include available models:

```yaml
# enterprise_knowledge_retriever/config.yaml
llm:
  "model": "gpt-oss-120b"
  "temperature": 0.0
  "max_tokens": 8192
  "available_models":  # ADD THIS SECTION
    - "gpt-oss-120b"
    - "Meta-Llama-3.3-70B-Instruct"
    - "DeepSeek-V3.1"
    - "your-custom-model-name"

embedding_model:
  "model": "E5-Mistral-7B-Instruct"

# ... rest of config
```

**Step 2**: Update Streamlit app to read from config:

```python
# In app.py
def load_config() -> Any:
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

config = load_config()

# Replace hardcoded LLM_MODELS with:
LLM_MODELS = config.get('llm', {}).get('available_models', [
    # Fallback defaults if not in config
    'gpt-oss-120b',
    'Meta-Llama-3.3-70B-Instruct',
])
```

This allows you to customize available models per kit without editing Python code!

**Apply this change to all apps** - see Appendix A for details.

#### **2.3.4 Environment Variables Setup**

**Edit your `.env` file**:

```bash
# Required: Your API Key
SAMBANOVA_API_KEY=your-api-key-from-custx-cloud

# Required for custom cloud: Your Cloud Base URL
SAMBANOVA_API_BASE=https://cloud.custx.ai/v1

# Optional: Third-party API keys (if using search tools)
SERPAPI_API_KEY=your-serpapi-key
TAVILY_API_KEY=your-tavily-key

# Optional: Analytics and tracing
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your-langsmith-key

# Production mode (hides environment variables in UI)
PROD_MODE=false
```

**Verification**:
```bash
# Check environment variables are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('SAMBANOVA_API_BASE'))"
```

### 2.4 Verification and Testing

#### **2.4.1 Pre-Deployment Checklist**

**Branding Checklist**:
- [ ] All logos replaced (dark, light, icon)
- [ ] All README files updated with your branding
- [ ] All URLs updated (cloud, community, website)
- [ ] Streamlit theme colors updated (all 9 apps)
- [ ] API key links updated in all Streamlit apps
- [ ] License and attribution preserved

**Configuration Checklist**:
- [ ] `.env` file created and configured
- [ ] `SAMBANOVA_API_BASE` set to your cloud URL
- [ ] API key valid and tested
- [ ] Model lists reviewed and customized
- [ ] Base URL input available in all Streamlit UIs

**Testing Checklist**:
- [ ] At least one Streamlit app launches successfully
- [ ] API connection works with custom base URL
- [ ] Models load and respond correctly
- [ ] Logos display correctly in UI
- [ ] No broken links in documentation
- [ ] No references to old branding remain

#### **2.4.2 Connection Testing Procedures**

**Test 1: Verify API Connectivity**

Create a test script `scripts/test_connection.py`:

```python
import os
from dotenv import load_dotenv
from langchain_sambanova import ChatSambaNova

load_dotenv()

def test_connection():
    api_key = os.getenv('SAMBANOVA_API_KEY')
    base_url = os.getenv('SAMBANOVA_API_BASE', 'https://api.sambanova.ai/v1')

    print(f"Testing connection to: {base_url}")
    print(f"API Key: {api_key[:10]}..." if api_key else "No API key found")

    try:
        llm = ChatSambaNova(
            model='gpt-oss-120b',
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )

        response = llm.invoke("Say 'Connection successful!' if you can read this.")
        print(f"\n✅ Success! Response: {response.content}")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == '__main__':
    test_connection()
```

**Run the test**:
```bash
python scripts/test_connection.py
```

**Test 2: Verify Each Streamlit App**

```bash
# Test Enterprise Knowledge Retriever
cd enterprise_knowledge_retriever
streamlit run streamlit/app.py

# Checklist while app is running:
# - Verify logo appears in sidebar
# - Verify page icon in browser tab
# - Check that base URL input field is visible
# - Test a simple query
# - Verify chat avatar uses your icon
# - Check that API key link points to your cloud URL
```

**Test 3: Verify Model Availability**

```python
# scripts/test_models.py
import os
from dotenv import load_dotenv
from langchain_sambanova import ChatSambaNova

load_dotenv()

models_to_test = [
    'gpt-oss-120b',
    'Meta-Llama-3.3-70B-Instruct',
    'DeepSeek-V3.1',
    # Add your custom models
]

for model in models_to_test:
    try:
        llm = ChatSambaNova(model=model)
        response = llm.invoke("Hi")
        print(f"✅ {model}: Working")
    except Exception as e:
        print(f"❌ {model}: {e}")
```

**Test 4: Link Validation**

```bash
# Install markdown-link-check
npm install -g markdown-link-check

# Check all links in READMEs
find . -name "README.md" -exec markdown-link-check {} \;
```

#### **2.4.3 Individual App Testing Guide**

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

### 2.5 Keeping Current with Updates

#### **Recommended Git Workflow**

```bash
# 1. Set up remotes
git remote add upstream https://github.com/sambanova/ai-starter-kit.git
git remote -v

# 2. Create a branch for upstream tracking
git checkout -b upstream-sync

# 3. Fetch upstream changes
git fetch upstream

# 4. Review changes
git log main..upstream/main --oneline
git diff main upstream/main

# 5. Merge or cherry-pick
git checkout main
git merge upstream/main
# Or selectively:
git cherry-pick <commit-hash>

# 6. Reapply customizations
./scripts/rebrand.sh  # Your rebranding script

# 7. Test
python scripts/test_connection.py
cd enterprise_knowledge_retriever && streamlit run streamlit/app.py

# 8. Commit
git add .
git commit -m "Merge upstream updates and reapply branding"
```

#### **Monitoring for Updates**

**Option 1: GitHub Watch**
- Click "Watch" on the SambaNova repository
- Select "Custom" → "Releases"

**Option 2: RSS Feed**
- Subscribe to: `https://github.com/sambanova/ai-starter-kit/releases.atom`

**Option 3: Automated Checks**

Create a GitHub Action (`.github/workflows/check-upstream.yml`):
```yaml
name: Check Upstream Updates

on:
  schedule:
    - cron: '0 0 * * 1'  # Every Monday
  workflow_dispatch:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check for upstream updates
        run: |
          git remote add upstream https://github.com/sambanova/ai-starter-kit.git
          git fetch upstream
          BEHIND=$(git rev-list --count HEAD..upstream/main)

          if [ $BEHIND -gt 0 ]; then
            echo "::warning::Your fork is $BEHIND commits behind upstream"
          fi
```

### 2.6 Support Information

#### **Internal Support Setup**

1. **Documentation**
   - Maintain a custom README_CUSTOM.md with your specific setup notes
   - Document any additional customizations
   - Keep a changelog of modifications

2. **Issue Tracking**
   - Use GitHub Issues in your forked repository
   - Tag issues: `branding`, `configuration`, `upstream-merge`, `bug`, `enhancement`

3. **Knowledge Base**
   - Document common issues and solutions
   - Create troubleshooting guides for your team
   - Maintain a FAQ

#### **When to Contact SambaNova**

Contact SambaNova support for:
- Issues with core functionality (not related to your customizations)
- Questions about model availability or performance
- API-related problems
- Security vulnerabilities

**Do not contact SambaNova for**:
- Issues introduced by your customizations
- Branding-related questions
- Custom code you've added

---

## Section 3: Agents Application Configuration

### 3.1 Overview of the Agents Application

The **SambaNova Agents** application is an advanced multi-agent AI system that:
- Intelligently routes requests to specialized agents
- Supports multiple LLM providers
- Provides a Vue 3 frontend with FastAPI backend
- Includes specialized subgraphs for financial analysis, research, and code execution

**Repository**: https://github.com/sambanova/agents

**Key Features**:
- Compound agent architecture
- WebSocket streaming
- Admin panel for provider configuration
- Support for custom LLM providers

### 3.2 Enabling the Admin Panel

By default, the Agents app uses SambaNova's configuration. To add custom providers, you need to enable the admin panel.

#### **Step 1: Clone the Agents Repository**

```bash
git clone https://github.com/sambanova/agents.git
cd agents
```

#### **Step 2: Configure Backend Environment**

Edit `backend/.env`:

```bash
# Copy example environment file
cp backend/.env.example backend/.env

# Edit the file
nano backend/.env
```

**Add/uncomment these lines**:

```bash
# Enable Admin Panel
SHOW_ADMIN_PANEL=true

# Authentication (required)
AUTH0_DOMAIN=your_auth0_domain
AUTH0_AUDIENCE=your_auth0_audience

# Allow users to provide their own API keys
ENABLE_USER_KEYS=true

# Redis encryption (required)
REDIS_MASTER_SALT=your-random-salt-here

# Optional: Third-party services
SERPER_KEY=your-serper-key
EXA_KEY=your-exa-key
TAVILY_API_KEY=your-tavily-key

# Optional: Monitoring
LANGSMITH_API_KEY=your-langsmith-key
```

#### **Step 3: Configure Frontend Environment**

Edit `frontend/sales-agent-crew/.env`:

```bash
# Copy example if it doesn't exist
cp frontend/sales-agent-crew/.env.example frontend/sales-agent-crew/.env

# Edit the file
nano frontend/sales-agent-crew/.env
```

**Add/update**:

```bash
# Enable Admin Panel in Frontend
VITE_SHOW_ADMIN_PANEL=true

# API Configuration
VITE_API_URL=/api
VITE_WEBSOCKET_URL=ws://localhost:8000

# Authentication (must match backend)
VITE_AUTH0_DOMAIN=your_auth0_domain
VITE_AUTH0_CLIENT_ID=your_auth0_client_id
VITE_AUTH0_AUDIENCE=your_auth0_audience
```

#### **Step 4: Start the Application**

**Backend**:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (in a new terminal):
```bash
cd frontend/sales-agent-crew
npm install
npm run dev
```

#### **Step 5: Verify Admin Panel Access**

1. Open the app in your browser: `http://localhost:5173`
2. Log in using Auth0
3. Look for the **Settings** or **Admin** icon (usually gear icon)
4. You should see options to configure LLM providers

### 3.3 Adding a Custom LLM Provider

Once the admin panel is enabled, you can add custom LLM providers through the UI.

#### **Step 1: Access Provider Settings**

1. Click the **Settings/Admin** icon in the app
2. Navigate to **LLM Providers** or **Model Configuration**
3. Click **Add Provider** or **Add Custom Provider**

#### **Step 2: Configure Provider Details**

You'll need to provide:

| Field | Description | Example |
|-------|-------------|---------|
| **Provider Name** | Display name for your provider | "CustomX Cloud" |
| **API Base URL** | Base endpoint for your cloud | `https://cloud.custx.ai/v1` |
| **API Key** | Your API authentication key | `sk-...` |
| **Models** | Available models (comma-separated or JSON) | `gpt-oss-120b, custom-model-1` |
| **Default Model** | The default model to use | `gpt-oss-120b` |

**Example Configuration (UI Form)**:

```
Provider Name: CustomX Cloud
API Base URL: https://cloud.custx.ai/v1
API Key: [your-api-key]
Available Models:
  - gpt-oss-120b
  - Meta-Llama-3.3-70B-Instruct
  - custom-model-1
Default Model: gpt-oss-120b
```

#### **Step 3: Test the Provider**

1. After saving, select your custom provider from the dropdown
2. Choose a model
3. Send a test message
4. Verify the response comes from your custom cloud

#### **Step 4: Configure as Default (Optional)**

In the admin panel:
1. Mark your custom provider as **Default**
2. All new sessions will use this provider
3. Users can still switch providers if needed

### 3.4 Advanced Configuration

#### **3.4.1 Provider Configuration via Environment**

If you prefer to configure providers via environment variables instead of the UI:

**Edit `backend/.env`**:

```bash
# Custom Provider Configuration
CUSTOM_PROVIDER_NAME=CustomX Cloud
CUSTOM_PROVIDER_BASE_URL=https://cloud.custx.ai/v1
CUSTOM_PROVIDER_API_KEY=your-api-key
CUSTOM_PROVIDER_MODELS=gpt-oss-120b,Meta-Llama-3.3-70B-Instruct,custom-model-1
```

**Note**: Check the Agents repository code to see if environment-based provider configuration is supported. If not, use the admin UI.

#### **3.4.2 User-Provided API Keys**

If `ENABLE_USER_KEYS=true`, users can provide their own API keys:

1. User clicks **Settings**
2. User enters their own API key
3. System uses user's key instead of environment key
4. Keys are encrypted with `REDIS_MASTER_SALT`

#### **3.4.3 Multiple Providers**

The Agents app supports multiple providers simultaneously:

1. Add multiple providers through the admin panel
2. Users can switch providers via dropdown
3. Each conversation can use a different provider

**Example setup**:
- Provider 1: Your Custom Cloud (default)
- Provider 2: OpenAI (for specific use cases)
- Provider 3: Azure OpenAI (for enterprise features)

#### **3.4.4 Branding the Agents App**

To white-label the Agents UI:

**Frontend Branding**:

1. **Logo**: Replace logo in `frontend/sales-agent-crew/src/assets/`
   ```bash
   cp /path/to/your-logo.png frontend/sales-agent-crew/src/assets/logo.png
   ```

2. **Colors**: Edit `frontend/sales-agent-crew/tailwind.config.js`
   ```javascript
   module.exports = {
     theme: {
       extend: {
         colors: {
           primary: '#0066CC',  // Your primary color
           secondary: '#003366', // Your secondary color
         }
       }
     }
   }
   ```

3. **Title**: Edit `frontend/sales-agent-crew/index.html`
   ```html
   <title>CustomX AI Agents</title>
   ```

4. **Metadata**: Update `frontend/sales-agent-crew/package.json`
   ```json
   {
     "name": "custx-ai-agents",
     "description": "CustomX AI Agent System",
     "author": "CustomX Inc."
   }
   ```

### 3.5 Verification and Testing

#### **Checklist**:
- [ ] Admin panel enabled and accessible
- [ ] Custom provider added successfully
- [ ] Test message sent and received
- [ ] Provider shows correct name and models
- [ ] API key securely stored
- [ ] Users can switch between providers (if multiple)
- [ ] Branding updated (logo, colors, title)

#### **Testing Procedure**:

**Test 1: Provider Connection**
```bash
# In backend directory
python -c "
from langchain_sambanova import ChatSambaNova
import os

llm = ChatSambaNova(
    base_url='https://cloud.custx.ai/v1',
    api_key='your-api-key',
    model='gpt-oss-120b'
)
print(llm.invoke('Hello!').content)
"
```

**Test 2: End-to-End Agent Flow**
1. Send a message that requires agent routing
2. Example: "What's the weather in San Francisco and analyze AAPL stock"
3. Verify correct agent activation
4. Check response quality and formatting

**Test 3: Model Switching**
1. Switch between different models in your provider
2. Send the same message to each model
3. Verify all models respond correctly

### 3.6 Troubleshooting

#### **Admin Panel Not Showing**

**Check**:
1. Both `SHOW_ADMIN_PANEL=true` and `VITE_SHOW_ADMIN_PANEL=true` are set
2. Backend and frontend restarted after environment changes
3. Auth0 configuration is correct
4. User has admin permissions (if role-based access is implemented)

**Debug**:
```bash
# Check backend logs
cd backend
uvicorn main:app --reload --log-level debug

# Check frontend console
# Open browser DevTools → Console tab
# Look for errors related to admin panel
```

#### **Provider Not Connecting**

**Check**:
1. API base URL is correct (include `/v1` if required)
2. API key is valid and not expired
3. Network can reach the URL (firewall, VPN)
4. Models specified are available on your cloud

**Debug**:
```bash
# Test API endpoint directly
curl -X POST https://cloud.custx.ai/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### **Models Not Appearing**

**Check**:
1. Model names match exactly (case-sensitive)
2. Models are deployed on your cloud
3. Provider configuration saved successfully
4. Frontend refreshed after saving

---

## Appendix A: Required Code Changes

This appendix lists all code changes required to fully support white labeling.

### A.1 Add Base URL Input to All Streamlit Apps

**Files to modify** (9 total):

1. `enterprise_knowledge_retriever/streamlit/app.py`
2. `multimodal_knowledge_retriever/streamlit/app.py`
3. `function_calling/streamlit/app.py`
4. `search_assistant/streamlit/app.py`
5. `benchmarking/streamlit/streamlit_utils.py`
6. `financial_assistant/streamlit/app.py`
7. `document_comparison/streamlit/app.py`
8. `eval_jumpstart/streamlit/app.py`
9. `data_extraction/streamlit/app.py` (if exists)

**Changes needed in each file**:

#### **Change 1: Update initialization**

**Find**:
```python
initialize_env_variables(config.get('prod_mode', False))
```

**Replace with**:
```python
initialize_env_variables(
    config.get('prod_mode', False),
    additional_env_vars=['SAMBANOVA_API_BASE']
)
```

#### **Change 2: Update credentials section**

**Find**:
```python
api_key, additional_vars = env_input_fields()
```

**Replace with**:
```python
api_key, additional_vars = env_input_fields(['SAMBANOVA_API_BASE'])
```

#### **Change 3: Pass base_url to API clients**

**Find** (example):
```python
llm = ChatSambaNova(
    model=model_name,
    api_key=sambanova_api_key,
    temperature=temperature,
)
```

**Replace with**:
```python
llm = ChatSambaNova(
    model=model_name,
    api_key=sambanova_api_key,
    base_url=st.session_state.get('SAMBANOVA_API_BASE'),
    temperature=temperature,
)
```

### A.2 Make Model Lists Configurable

**Files to modify** (9 total - same as above):

#### **Change 1: Update config.yaml files**

For each kit, edit its `config.yaml` to add:

```yaml
llm:
  "model": "gpt-oss-120b"
  "temperature": 0.0
  "max_tokens": 8192
  "available_models":  # ADD THIS
    - "gpt-oss-120b"
    - "Llama-4-Maverick-17B-128E-Instruct"
    - "Meta-Llama-3.3-70B-Instruct"
    - "DeepSeek-R1"
    - "DeepSeek-V3.1"
    - "Meta-Llama-3.1-8B-Instruct"
```

#### **Change 2: Update Python code to read from config**

**Find** (example):
```python
LLM_MODELS = [
    'gpt-oss-120b',
    'Llama-4-Maverick-17B-128E-Instruct',
    # ... more models
]
```

**Replace with**:
```python
# Load config first
config = load_config()

# Read models from config with fallback
LLM_MODELS = config.get('llm', {}).get('available_models', [
    'gpt-oss-120b',  # Fallback default
    'Meta-Llama-3.3-70B-Instruct',
])
```

### A.3 Update Hardcoded URLs

**Files to modify**: All Streamlit app.py files and streamlit_utils.py

**Find**:
```python
st.markdown('Get your SambaNova API key [here](https://cloud.sambanova.ai/apis)')
```

**Option 1: Replace with environment variable**:
```python
cloud_url = os.getenv('CLOUD_URL', 'https://cloud.sambanova.ai')
st.markdown(f'Get your API key [here]({cloud_url}/apis)')
```

**Option 2: Simple replacement**:
```python
st.markdown('Get your API key [here](https://cloud.custx.ai/apis)')
```

### A.4 Update Environment Utilities (Optional Enhancement)

**File**: `utils/visual/env_utils.py`

**Enhancement**: Make base URL input non-password field and add help text:

**Find** (line 66-68):
```python
additional_vars['SAMBANOVA_API_BASE'] = st.text_input(
    'SAMBANOVA API BASE',
    value=st.session_state.get('SAMBANOVA_API_BASE', ''),
    type='password'
)
```

**Replace with**:
```python
additional_vars['SAMBANOVA_API_BASE'] = st.text_input(
    'API Base URL',
    value=st.session_state.get('SAMBANOVA_API_BASE', 'https://api.sambanova.ai/v1'),
    help='The base URL for your cloud API endpoint (e.g., https://cloud.custx.ai/v1)'
)
```

### A.5 Summary of Changes

**Total files to modify**: ~20-25 files

**Breakdown**:
- Streamlit apps: 9 files
- Config YAML files: 9 files
- README files: 11 files
- Environment utilities: 1 file (optional)
- Logo files: 3 files (replacement, not modification)

**Estimated effort**:
- Code changes: 2-4 hours
- Testing: 2-3 hours
- Documentation updates: 1-2 hours
- **Total**: 5-9 hours for experienced developer

---

## Appendix B: Support & Resources

### B.1 Official SambaNova Resources

**Documentation**:
- SambaNova Docs: https://docs.sambanova.ai
- API Reference: https://docs.sambanova.ai/cloud/api-reference
- Integration Guides: https://docs.sambanova.ai/cloud/docs/integrations

**Community**:
- Community Forum: https://community.sambanova.ai
- GitHub Issues: https://github.com/sambanova/ai-starter-kit/issues
- GitHub Discussions: https://github.com/sambanova/ai-starter-kit/discussions

**Code Repositories**:
- AI Starter Kit: https://github.com/sambanova/ai-starter-kit
- Agents: https://github.com/sambanova/agents
- Integrations: https://github.com/sambanova/integrations

### B.2 Third-Party Tools & Resources

**Development Tools**:
- VS Code: https://code.visualstudio.com
- GitHub Copilot: https://github.com/features/copilot
- Streamlit: https://docs.streamlit.io

**Documentation Tools**:
- Markdown Guide: https://www.markdownguide.org
- Pandoc: https://pandoc.org
- MkDocs: https://www.mkdocs.org

**CI/CD & Automation**:
- GitHub Actions: https://docs.github.com/actions
- Pre-commit hooks: https://pre-commit.com

**Monitoring & Debugging**:
- LangSmith: https://docs.smith.langchain.com
- MLflow: https://mlflow.org

### B.3 Best Practices

#### **Version Control**
1. Always work in feature branches
2. Use descriptive commit messages
3. Tag releases (e.g., `v1.0.0-custx`)
4. Document changes in CHANGELOG.md

#### **Configuration Management**
1. Never commit `.env` files
2. Use `.env.example` as template
3. Document all configuration options
4. Use environment-specific configs (dev, staging, prod)

#### **Testing**
1. Test each component individually
2. Perform end-to-end integration tests
3. Validate all external links
4. Check UI responsiveness
5. Test with different API keys and endpoints

#### **Documentation**
1. Keep README files up to date
2. Document all customizations
3. Maintain a decision log
4. Include troubleshooting guides

#### **Security**
1. Rotate API keys regularly
2. Use environment variables for secrets
3. Enable authentication in production
4. Review code for security vulnerabilities
5. Keep dependencies updated

### B.4 Troubleshooting Common Issues

#### **Issue: "API Key Invalid" Error**

**Symptoms**: Applications fail to authenticate

**Solutions**:
1. Verify API key is correctly set in `.env`
2. Check for extra spaces or newlines in key
3. Ensure key hasn't expired
4. Verify key has correct permissions
5. Test key with curl:
   ```bash
   curl -H "Authorization: Bearer $SAMBANOVA_API_KEY" \
        https://cloud.custx.ai/v1/models
   ```

#### **Issue: "Module Not Found" Error**

**Symptoms**: Import errors when running apps

**Solutions**:
1. Ensure you're in correct directory
2. Activate virtual environment
3. Install requirements:
   ```bash
   pip install -r base-requirements.txt
   pip install -r [kit]/requirements.txt
   ```
4. Check Python version (3.10+ required)

#### **Issue: Streamlit App Won't Start**

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

#### **Issue: Logo Not Displaying**

**Symptoms**: Broken image icon in UI or README

**Solutions**:
1. Check file paths are correct
2. Verify image files exist and are readable
3. Check file permissions
4. Verify file format (PNG, SVG)
5. Clear browser cache

#### **Issue: Model Not Available**

**Symptoms**: "Model not found" or similar error

**Solutions**:
1. Check model name spelling (case-sensitive)
2. Verify model is deployed on your cloud
3. Check API endpoint supports the model:
   ```bash
   curl https://cloud.custx.ai/v1/models \
        -H "Authorization: Bearer $API_KEY"
   ```
4. Try a different model
5. Check model list configuration

### B.5 Getting Help

#### **Before Asking for Help**

1. Check this guide thoroughly
2. Review error messages carefully
3. Check logs (backend and frontend)
4. Search existing issues on GitHub
5. Verify your configuration is correct

#### **When Asking for Help**

Include:
1. Exact error message
2. Steps to reproduce
3. Environment details (OS, Python version)
4. Relevant configuration (redact secrets!)
5. What you've already tried

#### **Where to Get Help**

**For SambaNova-specific issues**:
- GitHub Issues: https://github.com/sambanova/ai-starter-kit/issues
- Community: https://community.sambanova.ai

**For your customizations**:
- Internal support channels
- Your development team
- This guide's troubleshooting section

---

## Document Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-13 | Initial release |

---

## License & Attribution

This guide is provided for SambaManaged customers. The original SambaNova AI Starter Kit and Agents application are licensed under their respective licenses. Ensure you comply with all license requirements when customizing and deploying these applications.

**Original Work**:
- AI Starter Kit: Copyright © SambaNova Systems
- Agents: Copyright © SambaNova Systems

**Customization Guide**:
- Copyright © 2025 [Your Company Name]

---

*End of White Labeling and Customization Guide*
