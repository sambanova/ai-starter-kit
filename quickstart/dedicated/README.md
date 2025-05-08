<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# API Examples - Dedicated Tier

Welcome to this section of API examples that are available exclusively to customers of the SambaNova Cloud dedicated tier. To run the example notebooks, you will need to set up the SambaNova SDK (SNSDK) and command line interface tool (SNAPI). 

1.  Install the following libraries using the wheels provided to you by your SambaNova representative:  
    - `pip install ~/Downloads/snsdk-<version>-py3-none-any.whl`  
    - `pip install ~/Downloads/SambaStudio_API_Client-<version>-py3-none-any.whl`

2.  Set the following environment variables in your `.env` file as provided by your SambaNova representative:
```  
SAMBASTUDIO_HOST_NAME = '<host name without any trailing "/">'
SAMBASTUDIO_ACCESS_KEY = '<access key>'
SAMBASTUDIO_TENANT_NAME = '<tenant name if provided, else "default">'
```

## Workflows Enabled

The notebooks in the folder demonstrate how the following workflows can be accomplished:
1. [Deploy an endpoint](<./Deploy a Model or Bundle to an Endpoint.ipynb>) with existing models or bundles.
2. [Create a new bundle](<./Create a Model Bundle.ipynb>) and then [deploy](<./Deploy a Model or Bundle to an Endpoint.ipynb>) it.
3. [Bring your own checkpoint](<./Bring Your Own Checkpoint (BYOC).ipynb>) and [deploy](<./Deploy a Model or Bundle to an Endpoint.ipynb>) it.
4. [Bring your own checkpoint](<./Bring Your Own Checkpoint (BYOC).ipynb>), [create a new bundle](<./Create a Model Bundle.ipynb>) with it, and then [deploy](<./Deploy a Model or Bundle to an Endpoint.ipynb>) the bundle.
