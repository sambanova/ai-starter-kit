<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Basic Examples
====================

# Overview

These are basic examples to get you started

# Setup instructions

1. Clone this repo:
```
git clone https://github.com/sambanova/ai-starter-kit.git 
```

2. Update `pip` and install dependencies. It is recommended to use a virtual env or `conda` environment for installation. For example: 
```
cd ai-starter-kit/workshops/ai_engineer_2024/basic_examples
conda create -n basic_ex python=3.10
conda activate basic_ex
pip  install  -r  requirements.txt
```

3. Create a .env file in ai-starket-kit (if it doesn't exist), and include the following info:
```
SAMBASTUDIO_BASE_URL="https://sjc3-demo2.sambanova.net"
SAMBASTUDIO_PROJECT_ID="bdadb40a-99d2-4705-96ff-48919435a0d8"
SAMBASTUDIO_ENDPOINT_ID="e0c07d60-3d7b-45f8-8653-088d4b8e8abd"
SAMBASTUDIO_API_KEY="d0838e80-e2c5-49f3-8c16-4127f97e9854"
```