import { NextRequest, NextResponse } from 'next/server';
import { execSync } from 'child_process';
import path from 'path';

/**
 * GET - Fetch last N lines of pod logs
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const podName = searchParams.get('podName');
    const lines = searchParams.get('lines') || '5';

    if (!podName || typeof podName !== 'string') {
      return NextResponse.json(
        { error: 'Pod name is required' },
        { status: 400 }
      );
    }

    const kubeconfigPath = path.join(process.cwd(), process.env.KUBECONFIG_FILE!);
    const namespace = process.env.NAMESPACE || 'default';
    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    try {
      // Run kubectl logs with tail
      const output = execSync(`kubectl -n ${namespace} logs ${podName} --tail=${lines}`, {
        encoding: 'utf-8',
        env,
        timeout: 10000, // 10 second timeout
      });

      return NextResponse.json({
        success: true,
        logs: output,
        podName,
      });
    } catch (error: any) {
      // Pod might not exist yet or kubectl command failed
      return NextResponse.json({
        success: false,
        error: 'Failed to fetch pod logs',
        message: error.message || 'Unknown error',
        stderr: error.stderr?.toString() || '',
        podName,
      });
    }
  } catch (error: any) {
    console.error('Pod logs error:', error);
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
