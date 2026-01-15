# Bundle Deployment Page

## Overview

The Bundle Deployment page manages the deployment lifecycle of validated bundles. It allows you to view existing deployments, create new deployments, monitor deployment status, and view pod logs in real-time.

## What Happens on This Page

### Section 1: Check Existing Bundle Deployments
- Lists all BundleDeployment resources in your namespace
- Shows deployment status (Deployed, Deploying, Not Deployed)
- Allows you to delete deployments
- Click "Status" to monitor a specific deployment

### Section 2: Deploy a Bundle
- Select from validated bundles
- Auto-generate deployment YAML
- Apply the deployment to your cluster
- Monitor deployment progress automatically

### Section 3: Check Deployment Status
- Real-time pod status monitoring (cache and default pods)
- Live log streaming from both pods
- Progress indicators showing container readiness
- Auto-refresh every 3 seconds

## kubectl Commands Used

All commands use the namespace configured on the Home page.

### 1. List Bundle Deployments
```bash
kubectl -n <namespace> get bundledeployment -o json
```
**Purpose**: Retrieves all BundleDeployment resources in the namespace
**When**: On page load and when clicking "Refresh"
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Lists all deployed bundles with their status and metadata

### 2. Get BundleDeployment Details
```bash
kubectl get bundledeployment <deployment-name> -n <namespace> -o json
```
**Purpose**: Retrieves detailed information about a specific deployment
**When**: When checking which models are deployed
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Gets the bundle reference and deployment spec

### 3. Get Bundle Details
```bash
kubectl get bundle <bundle-name> -n <namespace> -o json
```
**Purpose**: Retrieves the bundle specification including models
**When**: When fetching available models for the Playground
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Gets the list of models included in the bundle

### 4. List Valid Bundles
```bash
kubectl -n <namespace> get bundle -o json
```
**Purpose**: Lists all Bundle resources to show validated bundles available for deployment
**When**: On page load
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Retrieves all bundles and filters for those with validation status "ValidationSucceeded"

### 5. Apply BundleDeployment
```bash
kubectl -n <namespace> apply -f <bundle-deployment>.yaml
```
**Purpose**: Creates a new BundleDeployment resource
**When**: When you click "Deploy"
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Submits the deployment to the cluster, triggering pod creation

### 6. Delete BundleDeployment
```bash
kubectl -n <namespace> delete bundledeployment <deployment-name>
```
**Purpose**: Removes a BundleDeployment and its associated pods
**When**: When you click "Delete" on a deployment
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Tears down the deployment and all associated resources

### 7. Get Pod Status
```bash
kubectl -n <namespace> get pods | grep <deployment-name>
```
**Purpose**: Checks the status of pods associated with a deployment
**When**: Automatically every 3 seconds when monitoring a deployment
**Namespace**: Uses the namespace from your current environment configuration
**What It Does**: Shows readiness status (e.g., "1/1", "2/2") and pod state

### 8. Get Pod Logs (Cache Pod)
```bash
kubectl -n <namespace> logs <pod-name> --tail=5
```
**Purpose**: Retrieves the last 5 lines of logs from the cache pod
**When**: Automatically every 3 seconds when monitoring a deployment
**Namespace**: Uses the namespace from your current environment configuration
**Pod Name**: `inf-<deployment-name>-cache-0`

### 9. Get Pod Logs (Default Pod)
```bash
kubectl -n <namespace> logs <pod-name> -c inf --tail=5
```
**Purpose**: Retrieves the last 5 lines of logs from the default pod's inf container
**When**: Automatically every 3 seconds when monitoring a deployment
**Namespace**: Uses the namespace from your current environment configuration
**Pod Name**: `inf-<deployment-name>-q-default-n-0`

## Pod Architecture

Each BundleDeployment creates two main pods:

1. **Cache Pod** (`inf-<deployment-name>-cache-0`):
   - Loads model weights into memory
   - Serves as the model cache
   - Must be ready before inference can begin

2. **Default Pod** (`inf-<deployment-name>-q-default-n-0`):
   - Handles inference requests
   - Contains the `inf` container that processes queries
   - Requires cache pod to be ready first

## Deployment Status

- **Not Deployed**: Pods don't exist yet
- **Deploying**: Pods exist but containers are not all ready (e.g., "0/1", "1/2")
- **Deployed**: All pods are ready (e.g., "1/1", "2/2") and accepting requests
