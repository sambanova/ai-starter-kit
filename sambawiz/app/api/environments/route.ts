import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

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

export async function GET() {
  try {
    // Read app-config.json to get current configuration
    const configPath = path.join(process.cwd(), 'app-config.json');
    let environments: string[] = [];
    let defaultEnvironment: string | null = null;
    let defaultNamespace = 'default';
    let defaultApiKey = '';
    let defaultApiDomain = '';
    let defaultUiDomain = '';
    let checkpointsDir = '';
    let kubeconfigs: Record<string, KubeconfigEntry> = {};

    if (fs.existsSync(configPath)) {
      try {
        const configContent = fs.readFileSync(configPath, 'utf-8');
        const config: AppConfig = JSON.parse(configContent);

        // Get list of configured environments
        environments = Object.keys(config.kubeconfigs || {}).sort();

        // Get current environment and its settings
        defaultEnvironment = config.currentKubeconfig || null;
        checkpointsDir = config.checkpointsDir || '';
        kubeconfigs = config.kubeconfigs || {};

        // Get namespace, API key, and domains for current environment
        if (defaultEnvironment && config.kubeconfigs[defaultEnvironment]) {
          defaultNamespace = config.kubeconfigs[defaultEnvironment].namespace || 'default';
          defaultApiKey = config.kubeconfigs[defaultEnvironment].apiKey || '';
          defaultApiDomain = config.kubeconfigs[defaultEnvironment].apiDomain || '';
          defaultUiDomain = config.kubeconfigs[defaultEnvironment].uiDomain || '';
        }
      } catch (parseError) {
        console.error('Error parsing app-config.json:', parseError);
      }
    }

    return NextResponse.json({
      success: true,
      environments,
      defaultEnvironment,
      defaultNamespace,
      defaultApiKey,
      defaultApiDomain,
      defaultUiDomain,
      checkpointsDir,
      kubeconfigs
    });

  } catch (error) {
    console.error('Error fetching environments:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to fetch environments',
      environments: [],
      defaultEnvironment: null,
      defaultNamespace: 'default',
      defaultApiKey: '',
      checkpointsDir: ''
    }, { status: 500 });
  }
}
