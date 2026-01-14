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

interface PodStatus {
  cachePod: { ready: number; total: number; status: string } | null;
  defaultPod: { ready: number; total: number; status: string } | null;
}

/**
 * GET - Fetch pod status for a bundle deployment
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const deploymentName = searchParams.get('deploymentName');

    if (!deploymentName || typeof deploymentName !== 'string') {
      return NextResponse.json(
        { error: 'Deployment name is required' },
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

    try {
      // Run kubectl get pods and grep for the deployment name
      const output = execSync(`kubectl -n ${namespace} get pods | grep ${deploymentName}`, {
        encoding: 'utf-8',
        env,
        timeout: 10000, // 10 second timeout
      });

      // Parse the output to extract pod status
      const lines = output.trim().split('\n');
      const podStatus: PodStatus = {
        cachePod: null,
        defaultPod: null,
      };

      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length < 3) continue;

        const podName = parts[0];
        const readyStatus = parts[1]; // e.g., "1/1" or "1/2"
        const status = parts[2]; // e.g., "Running"

        // Parse ready status
        const [ready, total] = readyStatus.split('/').map(Number);

        if (podName.includes('-cache-')) {
          podStatus.cachePod = { ready, total, status };
        } else if (podName.includes('-q-default-n-')) {
          podStatus.defaultPod = { ready, total, status };
        }
      }

      return NextResponse.json({
        success: true,
        podStatus,
        deploymentName,
      });
    } catch (error: any) {
      // Pods might not exist yet or kubectl command failed
      return NextResponse.json({
        success: false,
        error: 'Failed to fetch pod status',
        message: error.message || 'Unknown error',
        stderr: error.stderr?.toString() || '',
        deploymentName,
      });
    }
  } catch (error: any) {
    console.error('Pod status error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error',
        message: error.message || 'Unknown error',
      },
      { status: 500 }
    );
  }
}
