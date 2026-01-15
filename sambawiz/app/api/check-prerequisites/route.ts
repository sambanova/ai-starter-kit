import { NextResponse } from 'next/server';
import { execSync } from 'child_process';

export async function GET() {
  const prerequisites = {
    kubectl: false,
    helm: false,
  };

  try {
    // Check if kubectl is installed
    try {
      execSync('kubectl version --client', { stdio: 'pipe' });
      prerequisites.kubectl = true;
    } catch (error) {
      console.error('kubectl not found');
    }

    // Check if helm is installed
    try {
      execSync('helm version', { stdio: 'pipe' });
      prerequisites.helm = true;
    } catch (error) {
      console.error('helm not found');
    }

    return NextResponse.json({
      success: true,
      prerequisites,
    });
  } catch (error) {
    console.error('Error checking prerequisites:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to check prerequisites',
      prerequisites,
    }, { status: 500 });
  }
}
