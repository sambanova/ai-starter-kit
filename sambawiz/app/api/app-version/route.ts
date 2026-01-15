import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

export async function GET() {
  try {
    const versionFilePath = path.join(process.cwd(), 'VERSION');

    if (!fs.existsSync(versionFilePath)) {
      return NextResponse.json({
        success: false,
        error: 'VERSION file not found',
      }, { status: 404 });
    }

    const versionContent = fs.readFileSync(versionFilePath, 'utf-8');
    const lines = versionContent.split('\n');

    let appVersion = null;
    for (const line of lines) {
      if (line.trim().startsWith('app:')) {
        appVersion = line.split(':')[1].trim();
        break;
      }
    }

    if (!appVersion) {
      return NextResponse.json({
        success: false,
        error: 'App version not found in VERSION file',
      }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      version: appVersion,
    });
  } catch (error) {
    console.error('Error reading app version:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to read app version',
    }, { status: 500 });
  }
}
