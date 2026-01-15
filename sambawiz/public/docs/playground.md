# Playground Page

## Overview

The Playground page provides an interactive chat interface to test and interact with your deployed models. It allows you to send prompts to models and see their responses along with performance metrics.

## What Happens on This Page

1. **Select Deployment**: Choose from deployed bundles (those with status "Deployed")
2. **Select Model**: Pick a specific model from the selected deployment
3. **Chat Interface**: Send messages and receive responses from the model
4. **Performance Metrics**: View tokens/second, total latency, and time to first token
5. **View Code**: Get code snippets for integrating with the API

## kubectl Commands Used

All commands use the namespace configured on the Home page.

### 1. List Bundle Deployments
```bash
kubectl -n <namespace> get bundledeployment -o json
```
**Purpose**: Retrieves all BundleDeployment resources to populate the deployment selector
**When**: On page load
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Lists all deployments and filters for those that are fully deployed

### 2. Get Pod Status (for filtering)
```bash
kubectl -n <namespace> get pods | grep <deployment-name>
```
**Purpose**: Checks if the deployment pods are ready before showing in the dropdown
**When**: For each deployment found
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Verifies that both cache and default pods are in "Running" state with all containers ready

### 3. Get BundleDeployment Details
```bash
kubectl get bundledeployment <deployment-name> -n <namespace> -o json
```
**Purpose**: Retrieves the bundle reference from the deployment
**When**: When a deployment is selected
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Gets the bundle name associated with the deployment

### 4. Get Bundle Models
```bash
kubectl get bundle <bundle-name> -n <namespace> -o json
```
**Purpose**: Retrieves the list of models in the bundle
**When**: After selecting a deployment
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Extracts the model names from the bundle spec.models field

## Chat Functionality

The Playground uses the SambaStack API for inference, not kubectl. The API calls use:
- **API Domain**: Configured on the Home page (e.g., `https://api.example.com`)
- **API Key**: Configured on the Home page for authentication
- **Model Name**: Selected from the dropdown (e.g., `Meta-Llama-3.1-8B-Instruct`)

### API Request Format
```
POST <apiDomain>/v1/chat/completions
Headers:
  Authorization: Bearer <apiKey>
  Content-Type: application/json
Body:
  {
    "model": "<selected-model>",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "<user-message>"},
      ...
    ]
  }
```

## Performance Metrics

The Playground displays the following metrics for each response:

- **Tokens/second**: Generation speed (higher is better)
- **Total Latency**: End-to-end response time in seconds
- **Time to First Token**: Latency before the first token is generated (lower is better)

## View Code Dialog

The "View Code" button shows Python and cURL examples for:
1. Calling the chat completions API
2. Using your configured API key and domain
3. Sending messages to the selected model

This helps developers integrate the deployed models into their applications.
