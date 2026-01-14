import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

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

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { encodedConfig, environmentName } = body;

    // Validation
    if (!encodedConfig || !environmentName) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Validate environment name has no whitespace
    if (/\s/.test(environmentName)) {
      return NextResponse.json(
        { success: false, error: 'Environment name cannot contain whitespaces' },
        { status: 400 }
      );
    }

    // Read current app-config.json
    const configPath = path.join(process.cwd(), 'app-config.json');
    let config: AppConfig;

    if (fs.existsSync(configPath)) {
      try {
        const configContent = fs.readFileSync(configPath, 'utf-8');
        config = JSON.parse(configContent);
      } catch (parseError) {
        return NextResponse.json(
          { success: false, error: 'Failed to parse app-config.json' },
          { status: 500 }
        );
      }
    } else {
      return NextResponse.json(
        { success: false, error: 'app-config.json not found' },
        { status: 500 }
      );
    }

    // Check if environment already exists
    if (config.kubeconfigs[environmentName]) {
      return NextResponse.json(
        { success: false, error: `Environment '${environmentName}' already exists` },
        { status: 400 }
      );
    }

    // Create temp directory if it doesn't exist
    const tempDir = path.join(process.cwd(), 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    // Create kubeconfigs directory if it doesn't exist
    const kubeconfigsDir = path.join(process.cwd(), 'kubeconfigs');
    if (!fs.existsSync(kubeconfigsDir)) {
      fs.mkdirSync(kubeconfigsDir, { recursive: true });
    }

    // Write encoded config to temp file
    const tempFilePath = path.join(tempDir, `${environmentName}_encoded.txt`);
    fs.writeFileSync(tempFilePath, encodedConfig);

    // Decode and save to kubeconfigs directory
    const kubeconfigFilePath = path.join(kubeconfigsDir, `${environmentName}.yaml`);

    try {
      await execAsync(`cat "${tempFilePath}" | base64 -d > "${kubeconfigFilePath}"`);
    } catch (decodeError) {
      // Clean up temp file
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      return NextResponse.json(
        { success: false, error: 'Failed to decode config. Please ensure it is valid base64.' },
        { status: 400 }
      );
    }

    // Clean up temp file
    if (fs.existsSync(tempFilePath)) {
      fs.unlinkSync(tempFilePath);
    }

    // Verify the decoded file exists and is not empty
    if (!fs.existsSync(kubeconfigFilePath) || fs.statSync(kubeconfigFilePath).size === 0) {
      return NextResponse.json(
        { success: false, error: 'Decoded config is empty or invalid' },
        { status: 400 }
      );
    }

    // Update app-config.json
    config.kubeconfigs[environmentName] = {
      file: `kubeconfigs/${environmentName}.yaml`,
      namespace: 'default',
      apiKey: ''
    };

    // Set as current kubeconfig
    config.currentKubeconfig = environmentName;

    // Write updated config back to file
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + '\n');

    return NextResponse.json({
      success: true,
      message: `Environment '${environmentName}' added successfully`
    });

  } catch (error) {
    console.error('Error adding environment:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 }
    );
  }
}
