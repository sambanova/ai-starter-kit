import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

interface KubeconfigEntry {
  file: string;
  namespace: string;
  apiKey?: string;
}

interface AppConfig {
  checkpointsDir: string;
  currentKubeconfig: string;
  kubeconfigs: Record<string, KubeconfigEntry>;
}

interface UpdateConfigRequest {
  environment: string;
  namespace: string;
  apiKey?: string;
}

export async function POST(request: Request) {
  try {
    const body: UpdateConfigRequest = await request.json();
    const { environment, namespace, apiKey } = body;

    if (!environment || !namespace) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields: environment and namespace'
      }, { status: 400 });
    }

    // Read existing config
    const configPath = path.join(process.cwd(), 'app-config.json');
    let config: AppConfig;

    if (fs.existsSync(configPath)) {
      const configContent = fs.readFileSync(configPath, 'utf-8');
      config = JSON.parse(configContent);
    } else {
      // Initialize with default values if config doesn't exist
      config = {
        checkpointsDir: '',
        currentKubeconfig: environment,
        kubeconfigs: {}
      };
    }

    // Check if the environment exists in kubeconfigs
    if (!config.kubeconfigs[environment]) {
      return NextResponse.json({
        success: false,
        error: `Environment '${environment}' not found in configuration`
      }, { status: 404 });
    }

    // Verify the kubeconfig file exists
    const kubeconfigFile = config.kubeconfigs[environment].file;
    const kubeconfigPath = path.join(process.cwd(), kubeconfigFile);
    if (!fs.existsSync(kubeconfigPath)) {
      return NextResponse.json({
        success: false,
        error: `Kubeconfig file not found: ${kubeconfigFile}`
      }, { status: 404 });
    }

    // Update current environment and namespace
    config.currentKubeconfig = environment;
    config.kubeconfigs[environment].namespace = namespace;

    // Update API key if provided
    if (apiKey !== undefined) {
      config.kubeconfigs[environment].apiKey = apiKey;
    }

    // Write updated config
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');

    return NextResponse.json({
      success: true,
      message: 'Configuration updated successfully',
      config: {
        environment,
        namespace,
        apiKey: apiKey !== undefined ? '***' : undefined,
        kubeconfigFile
      }
    });

  } catch (error) {
    console.error('Error updating configuration:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to update configuration'
    }, { status: 500 });
  }
}
