import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import path from 'path';

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
    const kubeconfigPath = path.join(process.cwd(), process.env.KUBECONFIG_FILE!);
    const namespace = process.env.NAMESPACE || 'default';
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
