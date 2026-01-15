import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import path from 'path';
import fs from 'fs';

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

/**
 * Compare two semantic versions treating each segment as an integer.
 * Returns: 1 if version1 > version2, -1 if version1 < version2, 0 if equal
 */
function compareVersions(version1: string, version2: string): number {
  const v1Parts = version1.split('.').map(Number);
  const v2Parts = version2.split('.').map(Number);
  const maxLength = Math.max(v1Parts.length, v2Parts.length);

  for (let i = 0; i < maxLength; i++) {
    const v1Part = v1Parts[i] || 0;
    const v2Part = v2Parts[i] || 0;

    if (v1Part > v2Part) return 1;
    if (v1Part < v2Part) return -1;
  }

  return 0;
}

/**
 * Read minimum required helm version from VERSION file
 */
function getMinimumHelmVersion(): string | null {
  try {
    const versionFilePath = path.join(process.cwd(), 'VERSION');
    if (!fs.existsSync(versionFilePath)) {
      console.error('VERSION file not found');
      return null;
    }

    const versionContent = fs.readFileSync(versionFilePath, 'utf-8');
    const lines = versionContent.split('\n');

    for (const line of lines) {
      if (line.trim().startsWith('minimum-sambastack-helm:')) {
        const version = line.split(':')[1].trim();
        return version;
      }
    }

    return null;
  } catch (error) {
    console.error('Error reading VERSION file:', error);
    return null;
  }
}

export async function GET() {
  try {
    // Read app-config.json to get current configuration
    const configPath = path.join(process.cwd(), 'app-config.json');
    let kubeconfigFile = '';
    let namespace = 'default';
    let envName = '';

    if (fs.existsSync(configPath)) {
      try {
        const configContent = fs.readFileSync(configPath, 'utf-8');
        const config: AppConfig = JSON.parse(configContent);

        const currentEnv = config.currentKubeconfig;
        if (currentEnv && config.kubeconfigs[currentEnv]) {
          kubeconfigFile = config.kubeconfigs[currentEnv].file;
          namespace = config.kubeconfigs[currentEnv].namespace || 'default';
          envName = currentEnv;
        }
      } catch (parseError) {
        console.error('Error parsing app-config.json:', parseError);
      }
    }

    if (!kubeconfigFile) {
      return NextResponse.json({
        success: false,
        error: 'No kubeconfig file configured',
        errorDetails: 'No kubeconfig file is configured in app-config.json. Please set up your environment configuration.',
        helmCommand: `helm list -n ${namespace} -o json`
      }, { status: 400 });
    }

    // Check if kubeconfig file exists
    const kubeconfigPath = path.join(process.cwd(), kubeconfigFile);

    if (!fs.existsSync(kubeconfigPath)) {
      return NextResponse.json({
        success: false,
        error: 'Kubeconfig file not found',
        errorDetails: `The kubeconfig file '${kubeconfigFile}' does not exist at path: ${kubeconfigPath}`,
        helmCommand: `helm list -n ${namespace} -o json`,
        notFound: true
      }, { status: 400 });
    }

    // Set KUBECONFIG environment variable
    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    // Run helm list command
    const helmOutput = execSync(`helm list -n ${namespace} -o json`, {
      env,
      timeout: 30000,
      encoding: 'utf-8'
    });

    // Parse the JSON output
    const helmReleases = JSON.parse(helmOutput);

    // Find the sambastack release
    const sambastackRelease = helmReleases.find(
      (release: { name: string }) => release.name === 'sambastack'
    );

    if (!sambastackRelease) {
      return NextResponse.json({
        success: false,
        error: 'sambastack release not found',
        errorDetails: 'The sambastack release was not found in the helm list output. Please ensure SambaStack is installed in the specified namespace.',
        helmCommand: `helm list -n ${namespace} -o json`
      }, { status: 404 });
    }

    // Extract chart version (e.g., "sambastack-0.3.496" -> "0.3.496")
    const chartField = sambastackRelease.chart;
    const version = chartField.replace('sambastack-', '');

    // Check if version meets minimum requirement
    const minimumVersion = getMinimumHelmVersion();
    if (minimumVersion) {
      const comparison = compareVersions(version, minimumVersion);
      if (comparison < 0) {
        return NextResponse.json({
          success: false,
          error: 'Helm version too old',
          errorDetails: `The installed sambastack helm chart version (${version}) is older than the minimum required version (${minimumVersion}). Please upgrade your sambastack installation.`,
          helmCommand: `helm list -n ${namespace} -o json`,
          version,
          minimumVersion,
          helmVersionError: true
        }, { status: 400 });
      }
    }

    return NextResponse.json({
      success: true,
      version,
      envName,
      namespace
    });

  } catch (error: unknown) {
    console.error('Kubeconfig validation error:', error);

    // Get the namespace for the error message
    let namespace = 'default';
    try {
      const configPath = path.join(process.cwd(), 'app-config.json');
      if (fs.existsSync(configPath)) {
        const configContent = fs.readFileSync(configPath, 'utf-8');
        const config: AppConfig = JSON.parse(configContent);
        const currentEnv = config.currentKubeconfig;
        if (currentEnv && config.kubeconfigs[currentEnv]) {
          namespace = config.kubeconfigs[currentEnv].namespace || 'default';
        }
      }
    } catch (e) {
      // Ignore error, use default namespace
    }

    // Determine error message and details
    let errorMessage = 'Failed to validate kubeconfig';
    let errorDetails = '';
    const helmCommand = `helm list -n ${namespace} -o json`;

    if (error instanceof Error) {
      errorDetails = error.message;
      // Include stderr output if available
      const execError = error as { stderr?: string };
      if (execError.stderr) {
        console.error('Helm command stderr:', execError.stderr);
        errorDetails = execError.stderr;
      }
    }

    return NextResponse.json({
      success: false,
      error: errorMessage,
      errorDetails,
      helmCommand
    }, { status: 500 });
  }
}
