import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import path from 'path';
import fs from 'fs';

export async function GET() {
  try {
    // Check if kubeconfig file exists
    const kubeconfigPath = path.join(process.cwd(), process.env.KUBECONFIG_FILE!);

    if (!fs.existsSync(kubeconfigPath)) {
      return NextResponse.json({
        success: false,
        error: 'Your kubeconfig.yaml seems to be invalid. Please check it and re-run the app. Also ensure that you are on the right network/VPN to access the server.',
        notFound: true
      }, { status: 400 });
    }

    // Set KUBECONFIG environment variable
    const namespace = process.env.NAMESPACE || 'default';
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
        error: 'sambastack release not found in helm list output'
      }, { status: 404 });
    }

    // Extract chart version (e.g., "sambastack-0.3.496" -> "0.3.496")
    const chartField = sambastackRelease.chart;
    const version = chartField.replace('sambastack-', '');

    // Extract environment name from kubeconfig filename
    // e.g., "kubeconfigs/sambastack-dev-2.yaml" -> "sambastack-dev-2"
    const kubeconfigFile = process.env.KUBECONFIG_FILE!;
    const filename = path.basename(kubeconfigFile, '.yaml');

    return NextResponse.json({
      success: true,
      version,
      envName: filename,
      namespace
    });

  } catch (error: unknown) {
    console.error('Kubeconfig validation error:', error);

    // Determine error message
    let errorMessage = 'Your kubeconfig.yaml seems to be invalid. Please check it and re-run the app. Also ensure that you are on the right network/VPN to access the server.';

    if (error instanceof Error) {
      // Include stderr output if available
      const execError = error as { stderr?: string };
      if (execError.stderr) {
        console.error('Helm command stderr:', execError.stderr);
      }
    }

    return NextResponse.json({
      success: false,
      error: errorMessage
    }, { status: 500 });
  }
}
