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

export async function POST() {
  try {
    const configPath = path.join(process.cwd(), 'app-config.json');
    const kubeconfigsDir = path.join(process.cwd(), 'kubeconfigs');

    if (!fs.existsSync(configPath)) {
      return NextResponse.json({
        success: false,
        error: 'app-config.json does not exist',
      }, { status: 400 });
    }

    if (!fs.existsSync(kubeconfigsDir)) {
      return NextResponse.json({
        success: false,
        error: 'kubeconfigs directory does not exist',
      }, { status: 400 });
    }

    // Read current config
    const configContent = fs.readFileSync(configPath, 'utf-8');
    const config: AppConfig = JSON.parse(configContent);

    // Check if kubeconfigs is empty and currentKubeconfig is empty
    const kubeconfigsEmpty = Object.keys(config.kubeconfigs || {}).length === 0;
    const currentKubeconfigEmpty = !config.currentKubeconfig || config.currentKubeconfig.trim() === '';

    if (!kubeconfigsEmpty || !currentKubeconfigEmpty) {
      return NextResponse.json({
        success: false,
        error: 'kubeconfigs is not empty or currentKubeconfig is already set',
      }, { status: 400 });
    }

    // Find all yaml files in kubeconfigs directory
    const files = fs.readdirSync(kubeconfigsDir);
    const yamlFiles = files.filter(
      (file) => (file.endsWith('.yaml') || file.endsWith('.yml')) && file !== 'kubeconfig_example.yaml'
    );

    if (yamlFiles.length === 0) {
      return NextResponse.json({
        success: false,
        error: 'No kubeconfig files found in kubeconfigs directory',
      }, { status: 400 });
    }

    // Auto-populate kubeconfigs
    config.kubeconfigs = {};
    yamlFiles.forEach((file) => {
      const envName = file.replace(/\.(yaml|yml)$/, '');
      config.kubeconfigs[envName] = {
        file: `kubeconfigs/${file}`,
        namespace: 'default',
      };
    });

    // Set the first one as current
    config.currentKubeconfig = yamlFiles[0].replace(/\.(yaml|yml)$/, '');

    // Save the updated config
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

    return NextResponse.json({
      success: true,
      message: 'kubeconfigs auto-populated successfully',
      config,
    });
  } catch (error) {
    console.error('Error auto-populating kubeconfigs:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to auto-populate kubeconfigs',
    }, { status: 500 });
  }
}
