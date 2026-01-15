import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';

const execAsync = promisify(exec);

interface KubeconfigEntry {
  file: string;
  namespace: string;
  apiKey?: string;
  apiDomain?: string;
  uiDomain?: string;
}

interface AppConfig {
  checkpointsDir: string;
  currentKubeconfig: string;
  kubeconfigs: Record<string, KubeconfigEntry>;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { environment } = body;

    if (!environment) {
      return NextResponse.json({
        success: false,
        error: 'Missing required field: environment'
      }, { status: 400 });
    }

    // Read app-config.json to get the kubeconfig and namespace
    const configPath = path.join(process.cwd(), 'app-config.json');
    if (!fs.existsSync(configPath)) {
      return NextResponse.json({
        success: false,
        error: 'Configuration file not found'
      }, { status: 404 });
    }

    const configContent = fs.readFileSync(configPath, 'utf-8');
    const config: AppConfig = JSON.parse(configContent);

    if (!config.kubeconfigs[environment]) {
      return NextResponse.json({
        success: false,
        error: `Environment '${environment}' not found in configuration`
      }, { status: 404 });
    }

    const kubeconfigEntry = config.kubeconfigs[environment];
    const namespace = kubeconfigEntry.namespace || 'default';
    const kubeconfigFile = path.join(process.cwd(), kubeconfigEntry.file);

    if (!fs.existsSync(kubeconfigFile)) {
      return NextResponse.json({
        success: false,
        error: `Kubeconfig file not found: ${kubeconfigEntry.file}`
      }, { status: 404 });
    }

    // Execute kubectl command to get keycloak credentials
    const command = `kubectl --kubeconfig="${kubeconfigFile}" -n ${namespace} get secret keycloak-initial-admin -o go-template='username: {{.data.username | base64decode}}{{\"\\n\"}}password: {{.data.password | base64decode}}{{\"\\n\"}}'`;

    try {
      const { stdout, stderr } = await execAsync(command);

      if (stderr && !stdout) {
        return NextResponse.json({
          success: false,
          error: 'Failed to retrieve credentials',
          details: stderr
        }, { status: 500 });
      }

      // Parse the output
      const lines = stdout.trim().split('\n');
      let username = '';
      let password = '';

      for (const line of lines) {
        if (line.startsWith('username: ')) {
          username = line.replace('username: ', '').trim();
        } else if (line.startsWith('password: ')) {
          password = line.replace('password: ', '').trim();
        }
      }

      if (!username || !password) {
        return NextResponse.json({
          success: false,
          error: 'Failed to parse credentials from kubectl output'
        }, { status: 500 });
      }

      return NextResponse.json({
        success: true,
        username,
        password
      });

    } catch (execError: any) {
      return NextResponse.json({
        success: false,
        error: 'Failed to execute kubectl command',
        details: execError.message
      }, { status: 500 });
    }

  } catch (error: any) {
    console.error('Error getting keycloak credentials:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to get keycloak credentials',
      details: error.message
    }, { status: 500 });
  }
}
