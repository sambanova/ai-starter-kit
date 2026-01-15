import { NextRequest, NextResponse } from 'next/server';
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
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

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const deploymentName = searchParams.get('deploymentName');

    if (!deploymentName) {
      return NextResponse.json(
        { success: false, error: 'deploymentName parameter is required' },
        { status: 400 }
      );
    }

    // Read app-config.json to get current kubeconfig and namespace
    const configPath = path.join(process.cwd(), 'app-config.json');
    if (!existsSync(configPath)) {
      return NextResponse.json(
        {
          success: false,
          error: 'app-config.json not found. Please configure an environment first.'
        },
        { status: 400 }
      );
    }

    const configContent = readFileSync(configPath, 'utf-8');
    const config: AppConfig = JSON.parse(configContent);

    const currentEnv = config.currentKubeconfig;
    if (!currentEnv || !config.kubeconfigs[currentEnv]) {
      return NextResponse.json(
        {
          success: false,
          error: 'No active environment configured. Please select an environment first.'
        },
        { status: 400 }
      );
    }

    const kubeconfigFile = config.kubeconfigs[currentEnv].file;
    const namespace = config.kubeconfigs[currentEnv].namespace || 'default';

    const kubeconfigPath = path.join(process.cwd(), kubeconfigFile);
    if (!existsSync(kubeconfigPath)) {
      return NextResponse.json(
        {
          success: false,
          error: `Kubeconfig file not found: ${kubeconfigFile}`
        },
        { status: 400 }
      );
    }

    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    // Step 1: Get the BundleDeployment to extract the bundle name
    let bundleDeploymentOutput: string;
    try {
      bundleDeploymentOutput = execSync(
        `kubectl get bundledeployment ${deploymentName} -n ${namespace} -o json`,
        {
          encoding: 'utf-8',
          env,
          timeout: 30000,
        }
      );
    } catch (error: any) {
      console.error('Error getting bundle deployment:', error);
      return NextResponse.json(
        {
          success: false,
          error: 'Failed to get bundle deployment',
          details: error.message,
          stderr: error.stderr?.toString() || '',
        },
        { status: 500 }
      );
    }

    // Parse the BundleDeployment JSON
    let bundleDeployment: any;
    try {
      bundleDeployment = JSON.parse(bundleDeploymentOutput);
    } catch (error: any) {
      console.error('Error parsing bundle deployment JSON:', error);
      return NextResponse.json(
        {
          success: false,
          error: 'Failed to parse bundle deployment data',
          details: error.message,
        },
        { status: 500 }
      );
    }

    // Extract bundle name from spec
    const bundleName = bundleDeployment?.spec?.bundle;
    if (!bundleName) {
      return NextResponse.json(
        {
          success: false,
          error: 'Bundle name not found in deployment spec',
        },
        { status: 404 }
      );
    }

    // Step 2: Get the Bundle to extract the models
    let bundleOutput: string;
    try {
      bundleOutput = execSync(
        `kubectl get bundle ${bundleName} -n ${namespace} -o json`,
        {
          encoding: 'utf-8',
          env,
          timeout: 30000,
        }
      );
    } catch (error: any) {
      console.error('Error getting bundle:', error);
      return NextResponse.json(
        {
          success: false,
          error: 'Failed to get bundle',
          details: error.message,
          stderr: error.stderr?.toString() || '',
        },
        { status: 500 }
      );
    }

    // Parse the Bundle JSON
    let bundle: any;
    try {
      bundle = JSON.parse(bundleOutput);
    } catch (error: any) {
      console.error('Error parsing bundle JSON:', error);
      return NextResponse.json(
        {
          success: false,
          error: 'Failed to parse bundle data',
          details: error.message,
        },
        { status: 500 }
      );
    }

    // Extract model names from spec.models
    const models = bundle?.spec?.models;
    if (!models || typeof models !== 'object') {
      return NextResponse.json(
        {
          success: false,
          error: 'Models not found in bundle spec',
        },
        { status: 404 }
      );
    }

    // Get model names (keys) and sort alphabetically (case-insensitive)
    const modelNames = Object.keys(models).sort((a, b) =>
      a.toLowerCase().localeCompare(b.toLowerCase())
    );

    return NextResponse.json({
      success: true,
      deploymentName,
      bundleName,
      models: modelNames,
    });
  } catch (error: any) {
    console.error('Unexpected error in deployment-models API:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'An unexpected error occurred',
        details: error.message,
      },
      { status: 500 }
    );
  }
}
