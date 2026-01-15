import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const kubeconfigsDir = path.join(process.cwd(), 'kubeconfigs');

    if (!fs.existsSync(kubeconfigsDir)) {
      return NextResponse.json({
        success: true,
        hasFiles: false,
        message: 'kubeconfigs directory does not exist',
      });
    }

    const files = fs.readdirSync(kubeconfigsDir);
    const yamlFiles = files.filter(
      (file) => (file.endsWith('.yaml') || file.endsWith('.yml')) && file !== 'kubeconfig_example.yaml'
    );

    return NextResponse.json({
      success: true,
      hasFiles: yamlFiles.length > 0,
      fileCount: yamlFiles.length,
      files: yamlFiles,
    });
  } catch (error) {
    console.error('Error checking kubeconfig files:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to check kubeconfig files',
    }, { status: 500 });
  }
}
