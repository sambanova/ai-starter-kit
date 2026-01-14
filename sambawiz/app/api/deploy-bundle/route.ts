import { NextRequest, NextResponse } from 'next/server';
import { writeFileSync } from 'fs';
import { execSync } from 'child_process';
import path from 'path';

/**
 * POST - Deploy a bundle by applying BundleDeployment YAML
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { yaml } = body;

    if (!yaml || typeof yaml !== 'string') {
      return NextResponse.json(
        { error: 'Invalid YAML content' },
        { status: 400 }
      );
    }

    // Save YAML to temporary file
    const timestamp = Date.now();
    const fileName = `bundle-deployment-${timestamp}.yaml`;
    const filePath = path.join(process.cwd(), 'temp', fileName);

    // Ensure temp directory exists
    const tempDir = path.join(process.cwd(), 'temp');
    try {
      execSync(`mkdir -p "${tempDir}"`);
    } catch (error) {
      // Directory might already exist
    }

    // Write YAML to file
    writeFileSync(filePath, yaml, 'utf-8');

    // Set KUBECONFIG environment variable
    const kubeconfigPath = path.join(process.cwd(), process.env.KUBECONFIG_FILE!);
    const namespace = process.env.NAMESPACE || 'default';
    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    let applyOutput = '';
    try {
      // Run kubectl apply
      applyOutput = execSync(`kubectl -n ${namespace} apply -f "${filePath}"`, {
        encoding: 'utf-8',
        env,
        timeout: 30000, // 30 second timeout
      });
    } catch (error: any) {
      // kubectl apply failed
      return NextResponse.json(
        {
          success: false,
          error: 'kubectl apply failed',
          message: error.message || 'Unknown error',
          stderr: error.stderr?.toString() || '',
          stdout: error.stdout?.toString() || '',
          filePath,
        },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Bundle deployment applied successfully',
      output: applyOutput.trim(),
      filePath,
    });
  } catch (error: any) {
    console.error('Deployment error:', error);
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
