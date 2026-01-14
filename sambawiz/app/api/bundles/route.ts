import { NextResponse } from 'next/server';
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

interface Bundle {
  name: string;
  namespace: string;
  template: string;
  creationTimestamp: string;
  isValid: boolean;
  validationReason: string;
  validationMessage: string;
  models: { [key: string]: any };
}

/**
 * GET - Fetch all bundles
 */
export async function GET() {
  try {
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

    const output = execSync(`kubectl -n ${namespace} get bundle -o json`, {
      encoding: 'utf-8',
      env,
      timeout: 30000,
    });

    const data = JSON.parse(output);

    // Transform the data to a more usable format
    const bundles: Bundle[] = data.items.map((item: any) => {
      const condition = item.status?.conditions?.[0];
      const isValid = condition?.reason === 'ValidationSucceeded' || condition?.status === 'True';

      return {
        name: item.metadata.name,
        namespace: item.metadata.namespace,
        template: item.spec.template,
        creationTimestamp: item.metadata.creationTimestamp,
        isValid,
        validationReason: condition?.reason || 'Unknown',
        validationMessage: condition?.message || '',
        models: item.spec.models || {},
      };
    });

    return NextResponse.json({
      success: true,
      bundles,
    });
  } catch (error: any) {
    console.error('Failed to fetch bundles:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch bundles',
        message: error.message || 'Unknown error',
        stderr: error.stderr?.toString() || '',
      },
      { status: 500 }
    );
  }
}
