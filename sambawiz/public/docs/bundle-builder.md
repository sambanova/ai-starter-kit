# Bundle Builder Page

## Overview

The Bundle Builder page allows you to create and configure SambaStack model bundles. A bundle is a collection of compiled model PEFs (Processor Executable Format) that can be deployed to your Kubernetes cluster.

## What Happens on This Page

1. **Model Selection**: Choose one or more models from the available model catalog
2. **Configuration Selection**: For each model, select PEF configurations (Sequence Size × Batch Size combinations)
3. **Speculative Decoding**: Optionally configure draft models for speculative decoding to improve performance
4. **YAML Generation**: Automatically generates BundleTemplate and Bundle YAML manifests
5. **Validation**: Validate the bundle by applying it to your cluster
6. **Save**: Save the generated YAML to the `saved_artifacts/` directory
7. **Create Deployment**: Navigate directly to deployment after successful validation

## kubectl Commands Used

All commands use the namespace configured on the Home page.

### 1. List Available Models
```bash
kubectl -n <namespace> get models
```
**Purpose**: Lists all models that are available for bundling
**What It Does**: Displays the models available in your SambaStack environment

### 2. List Available PEFs
```bash
kubectl -n <namespace> get pefs
```
**Purpose**: Lists all PEFs available for a given model, based on batch size and sequence length
**What It Does**: Shows the pre-compiled PEF configurations you can use in your bundles

### 3. Apply Bundle YAML
```bash
kubectl -n <namespace> apply -f <temp-bundle-file>.yaml
```
**Purpose**: Creates or updates the BundleTemplate and Bundle resources in the cluster
**When**: When you click "Validate"
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Submits the bundle YAML for validation by the SambaStack operator

### 4. Get Bundle Status
```bash
kubectl -n <namespace> get bundle <bundle-name> -o json
```
**Purpose**: Checks the validation status of the bundle
**When**: After applying the bundle (5 seconds wait)
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Retrieves the bundle status to determine if validation succeeded or failed

## Key Concepts

### PEF (Processor Executable Format)
Pre-compiled model executables optimized for specific sequence sizes (SS) and batch sizes (BS). Each model can have multiple PEF configurations.

### Speculative Decoding
A performance optimization technique where a smaller "draft" model generates candidate tokens that a larger "target" model validates. This can significantly improve inference speed.

### Bundle vs BundleTemplate
- **BundleTemplate**: Defines the models and their configurations
- **Bundle**: An instance of a BundleTemplate with resolved checkpoint paths

## Validation Process

1. Generate YAML with BundleTemplate and Bundle resources
2. Apply YAML to cluster using kubectl
3. SambaStack operator validates:
   - Checkpoint paths exist and are accessible
   - PEF configurations are compatible
   - Resource requirements can be met
4. Display validation result to user
