# SambaWiz

SambaWiz is a GUI wizard that accelerates the creation and deployment of model bundles on [SambaStack](https://docs.sambanova.ai/docs/en/admin/overview/sambastack-overview).

## Overview

SambaWiz provides an intuitive interface to:
- Select AI models from an available catalog
- Configure PEF (Processor Executable Format) settings including sequence size (SS) and batch size (BS)
- Map models to checkpoints
- Generate valid Kubernetes YAML manifests (BundleTemplate and Bundle resources)
- Validate and apply bundles to a Kubernetes cluster
- View bundle validation status and error messages

## Prerequisites

- Access to a Kubernetes cluster with SambaStack [installed](https://docs.sambanova.ai/docs/en/admin/installation/prerequisites) (minimum helm version of `0.3.496`)
- Valid `kubeconfig.yaml` for your SambaStack environment
- Node.js 18+ and npm
- `kubectl` and `helm` CLI tools installed and configured

## Features

- **Model Selection**: Choose from multiple AI models including Meta-Llama, DeepSeek, Qwen, and more
- **PEF Configuration**: Configure sequence sizes (16k, 32k, etc.) and batch sizes for each model
- **Automatic Checkpoint Mapping**: Models are automatically mapped to their corresponding checkpoints
- **YAML Generation**: Generates properly formatted Kubernetes manifests with:
  - BundleTemplate with expert configurations
  - Bundle with checkpoint mappings
  - Checkpoint sharing UUIDs for resource optimization
- **Editable YAML**: Manually edit generated YAML before validation
- **Kubernetes Integration**: Validate bundles by applying them to your cluster via `kubectl`
- **Real-time Status**: View bundle validation status and detailed error messages
- **Environment Version Display**: Automatically validates kubeconfig and displays SambaStack Helm version in the navigation sidebar


## Getting Started

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Application Settings

Create an `app-config.json` file in the project root directory by copying the example:

```bash
# Copy the example config file
cp app-config.example.json app-config.json
```

Edit `app-config.json` with your settings:

```json
{
  "checkpointsDir": "gs://your-bucket-name/path/to/checkpoints/",
  "currentKubeconfig": "your-environment-name",
  "kubeconfigs": {
    "your-environment-name": {
      "file": "kubeconfigs/your-environment.yaml",
      "namespace": "default",
      "apiKey": ""
    }
  }
}
```

**Important**:
- `app-config.json` is gitignored for security
- `checkpointsDir`: GCS checkpoint directory path (must end with `/`)
- `currentKubeconfig`: Name of the currently selected environment
- `kubeconfigs`: Object containing all configured environments
  - Each environment has:
    - `file`: Path to kubeconfig file relative to sambawiz folder
    - `namespace`: Kubernetes namespace for this environment
    - `apiKey`: Optional API key for environment-specific authentication
- The checkpoints directory is used to construct full checkpoint paths
- Configuration can be updated through the home page UI
- You can configure multiple environments in the `kubeconfigs` object

### 3. Configure Kubernetes Access

Place your kubeconfig files in the `kubeconfigs/` directory:

```bash
# Copy your kubeconfig to the kubeconfigs directory
cp /path/to/your/kubeconfig.yaml ./kubeconfigs/your-environment.yaml
```

Then add the environment to the `kubeconfigs` object in `app-config.json` with the corresponding file path, namespace, and optional API key.

**Important**:
- All files in the `kubeconfigs/` directory are gitignored for security (except `kubeconfig_example.yaml`)
- The application reads the kubeconfig file path from `app-config.json`
- The kubeconfig is validated on app startup using `helm list` to verify cluster connectivity
- If validation fails, an error alert is displayed with instructions to check your kubeconfig and network/VPN connection
- The SambaStack Helm version is displayed in the navigation sidebar when validation succeeds

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

The home page will display the environment selector where you can choose your kubeconfig and namespace.

### 5. Build for Production

```bash
npm run build
npm start
```

## Usage

1. **Select Models**: Choose one or more AI models from the dropdown
2. **Configure PEFs**: For each model, select desired SS/BS combinations from the configuration table
3. **Set Bundle Name**: Edit the auto-generated bundle name if desired
4. **Review YAML**: The YAML manifest is generated automatically and can be edited manually
5. **Validate**: Click "Validate" to apply the bundle to your Kubernetes cluster
6. **View Status**: See the validation results including any error messages from the cluster

## Project Structure

```
sambawiz/
├── app/
│   ├── api/
│   │   ├── kubeconfig-validate/    # API endpoint for kubeconfig validation
│   │   └── validate/               # API endpoint for bundle validation
│   ├── components/
│   │   ├── AppLayout.tsx           # Main layout with navigation and version display
│   │   └── BundleForm.tsx          # Main form component
│   ├── data/
│   │   ├── pef_configs.json        # PEF configuration data
│   │   ├── pef_mapping.json        # Model to PEF mappings
│   │   └── checkpoint_mapping.json # Model to checkpoint mappings
│   ├── utils/
│   │   └── bundle-yaml-generator.ts # YAML generation logic
│   ├── lib/
│   │   └── emotion-cache.ts        # MUI styling cache
│   ├── types/
│   │   └── bundle.ts               # TypeScript interfaces
│   ├── theme.ts                    # MUI theme configuration
│   └── page.tsx                    # Home page
├── kubeconfigs/                    # Kubeconfig files (gitignored except example)
│   ├── your-kubeconfig-name.yaml   # Your kubeconfig (gitignored)
│   └── kubeconfig_example.yaml     # Example template
├── scripts/                        # Utility scripts
│   ├── generate_pef_configs.ts     # Script to regenerate PEF configs
│   └── test_parsing.ts             # Testing utilities
├── public/                         # Static assets
└── temp/                           # Temporary YAML files (gitignored)
```

## Configuration Files

The application uses three JSON configuration files in the `app/data/` directory:

- **pef_configs.json**: Defines PEF configurations with SS, BS, and version info
- **pef_mapping.json**: Maps model names to their available PEF configurations
- **checkpoint_mapping.json**: Maps model names to their checkpoint GCS paths

## API Endpoints

### GET /api/kubeconfig-validate

Validates kubeconfig and retrieves SambaStack Helm version.

**Response (Success):**
```json
{
  "success": true,
  "version": "0.3.496"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Your kubeconfig.yaml seems to be invalid. Please check it and re-run the app. Also ensure that you are on the right network/VPN to access the server."
}
```

### POST /api/validate

Validates and applies a bundle YAML to the Kubernetes cluster.

**Request Body:**
```json
{
  "yaml": "apiVersion: sambanova.ai/v1alpha1\nkind: BundleTemplate\n..."
}
```

**Response:**
```json
{
  "success": true,
  "message": "Bundle validated and applied successfully",
  "applyOutput": "bundletemplate.sambanova.ai/bt-name created\nbundle.sambanova.ai/b-name created",
  "statusConditions": "Last Transition Time: ...\nMessage: ...",
  "bundleName": "b-name",
  "filePath": "/path/to/temp/bundle-123456.yaml"
}
```

## Technology Stack

- **Framework**: Next.js 15 (App Router)
- **UI Library**: Material-UI (MUI) v6
- **Language**: TypeScript
- **Styling**: Emotion (CSS-in-JS)
- **Backend**: Next.js API Routes with Node.js child_process for kubectl

## Development

```bash
# Run development server with hot reload
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build
```

## Notes

- The application requires a valid Kubernetes cluster with SambaNova CRDs installed
- Bundle validation involves applying resources to the cluster and checking their status
- Temporary YAML files are stored in the `temp/` directory
- The application uses `kubectl` commands via Node.js `execSync`, so ensure kubectl is in your PATH

## Security Considerations

- `app-config.json` and all files in `kubeconfigs/` (except the example) are gitignored to prevent credential leaks
- Temporary YAML files in `temp/` are also gitignored
- The validation endpoint runs kubectl commands server-side with appropriate timeouts
- Kubeconfig validation is performed on app startup to ensure cluster connectivity
- Consider implementing authentication/authorization for production deployments
- Never commit sensitive configuration files or credentials to version control
- Use `app-config.example.json` as a template (safe to commit)

## License

This project is part of the SambaNova AI Starter Kit.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
