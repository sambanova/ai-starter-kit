import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

interface KubeconfigEntry {
  file: string;
  namespace: string;
  uiDomain?: string;
  apiDomain?: string;
  apiKey?: string;
}

interface AppConfig {
  checkpointsDir: string;
  currentKubeconfig: string;
  kubeconfigs: Record<string, KubeconfigEntry>;
}

export async function GET() {
  try {
    const configPath = path.join(process.cwd(), 'app-config.json');
    const configExists = fs.existsSync(configPath);

    if (!configExists) {
      return NextResponse.json({
        success: true,
        exists: false,
        valid: false,
        message: 'app-config.json does not exist',
      });
    }

    // Check if the file is valid JSON and has checkpointsDir
    try {
      const configContent = fs.readFileSync(configPath, 'utf-8');
      const config: AppConfig = JSON.parse(configContent);

      const hasCheckpointsDir = config.checkpointsDir && config.checkpointsDir.trim() !== '';

      return NextResponse.json({
        success: true,
        exists: true,
        valid: hasCheckpointsDir,
        config,
        message: hasCheckpointsDir ? 'Configuration is valid' : 'checkpointsDir is not populated',
      });
    } catch (parseError) {
      return NextResponse.json({
        success: true,
        exists: true,
        valid: false,
        message: 'app-config.json is not valid JSON',
      });
    }
  } catch (error) {
    console.error('Error checking app-config.json:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to check app-config.json',
    }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const { checkpointsDir } = await request.json();

    if (!checkpointsDir || checkpointsDir.trim() === '') {
      return NextResponse.json({
        success: false,
        error: 'checkpointsDir is required',
      }, { status: 400 });
    }

    const configPath = path.join(process.cwd(), 'app-config.json');
    const kubeconfigsDir = path.join(process.cwd(), 'kubeconfigs');

    // Create minimal app-config.json
    const config: AppConfig = {
      checkpointsDir,
      currentKubeconfig: '',
      kubeconfigs: {},
    };

    // Check if there are any kubeconfig files in the kubeconfigs directory
    if (fs.existsSync(kubeconfigsDir)) {
      const files = fs.readdirSync(kubeconfigsDir);
      const yamlFiles = files.filter(
        (file) => (file.endsWith('.yaml') || file.endsWith('.yml')) && file !== 'kubeconfig_example.yaml'
      );

      if (yamlFiles.length > 0) {
        // Auto-populate kubeconfigs
        yamlFiles.forEach((file) => {
          const envName = file.replace(/\.(yaml|yml)$/, '');
          config.kubeconfigs[envName] = {
            file: `kubeconfigs/${file}`,
            namespace: 'default',
          };
        });

        // Set the first one as current
        config.currentKubeconfig = yamlFiles[0].replace(/\.(yaml|yml)$/, '');
      }
    }

    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

    return NextResponse.json({
      success: true,
      message: 'app-config.json created successfully',
      config,
    });
  } catch (error) {
    console.error('Error creating app-config.json:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to create app-config.json',
    }, { status: 500 });
  }
}
