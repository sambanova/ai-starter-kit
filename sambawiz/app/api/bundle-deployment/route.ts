import { NextRequest, NextResponse } from 'next/server';
import { execSync } from 'child_process';
import path from 'path';

interface BundleDeployment {
  name: string;
  namespace: string;
  bundle: string;
  creationTimestamp: string;
  status?: {
    conditions?: Array<{
      type: string;
      status: string;
      reason: string;
      message: string;
    }>;
  };
}

/**
 * GET - Fetch all bundle deployments
 */
export async function GET() {
  try {
    const kubeconfigPath = path.join(process.cwd(), process.env.KUBECONFIG_FILE!);
    const namespace = process.env.NAMESPACE || 'default';
    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    const output = execSync(`kubectl -n ${namespace} get bundledeployment -o json`, {
      encoding: 'utf-8',
      env,
      timeout: 30000,
    });

    const data = JSON.parse(output);

    // Transform the data to a more usable format
    const bundleDeployments: BundleDeployment[] = data.items.map((item: any) => ({
      name: item.metadata.name,
      namespace: item.metadata.namespace,
      bundle: item.spec.bundle,
      creationTimestamp: item.metadata.creationTimestamp,
      status: item.status,
    }));

    return NextResponse.json({
      success: true,
      bundleDeployments,
    });
  } catch (error: any) {
    console.error('Failed to fetch bundle deployments:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch bundle deployments',
        message: error.message || 'Unknown error',
        stderr: error.stderr?.toString() || '',
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE - Delete a bundle deployment
 */
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { name } = body;

    if (!name || typeof name !== 'string') {
      return NextResponse.json(
        { error: 'Bundle deployment name is required' },
        { status: 400 }
      );
    }

    const kubeconfigPath = path.join(process.cwd(), process.env.KUBECONFIG_FILE!);
    const namespace = process.env.NAMESPACE || 'default';
    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    const output = execSync(`kubectl -n ${namespace} delete bundledeployment ${name}`, {
      encoding: 'utf-8',
      env,
      timeout: 30000,
    });

    return NextResponse.json({
      success: true,
      message: `Bundle deployment ${name} deleted successfully`,
      output: output.trim(),
    });
  } catch (error: any) {
    console.error('Failed to delete bundle deployment:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete bundle deployment',
        message: error.message || 'Unknown error',
        stderr: error.stderr?.toString() || '',
      },
      { status: 500 }
    );
  }
}
