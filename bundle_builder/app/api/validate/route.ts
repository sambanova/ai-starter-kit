import { NextRequest, NextResponse } from 'next/server';
import { writeFileSync } from 'fs';
import { execSync } from 'child_process';
import path from 'path';

/**
 * Extract bundle name from YAML content
 */
function extractBundleName(yaml: string): string | null {
  const lines = yaml.split('\n');
  let inBundleSection = false;

  for (const line of lines) {
    // Check if we're in the Bundle section (not BundleTemplate)
    if (line.includes('kind: Bundle') && !line.includes('kind: BundleTemplate')) {
      inBundleSection = true;
      continue;
    }

    // Look for metadata.name in the Bundle section
    if (inBundleSection && line.includes('name:')) {
      const match = line.match(/name:\s*(.+)/);
      if (match) {
        return match[1].trim();
      }
    }

    // Reset if we hit another resource separator
    if (line.trim() === '---') {
      inBundleSection = false;
    }
  }

  return null;
}

/**
 * Extract validation status from kubectl JSON output
 */
interface BundleCondition {
  type: string;
  status: string;
  reason: string;
  message: string;
  lastTransitionTime?: string;
  observedGeneration?: number;
}

interface BundleStatus {
  status?: {
    conditions?: BundleCondition[];
  };
}

function extractValidationStatus(jsonOutput: string): {
  reason: string;
  message: string;
  isValid: boolean;
} {
  try {
    const bundle: BundleStatus = JSON.parse(jsonOutput);

    if (!bundle.status?.conditions || bundle.status.conditions.length === 0) {
      return {
        reason: 'Unknown',
        message: 'No validation conditions found in bundle status',
        isValid: false,
      };
    }

    // Get the first condition (typically the validation result)
    const condition = bundle.status.conditions[0];

    return {
      reason: condition.reason || 'Unknown',
      message: condition.message || 'No message provided',
      isValid: condition.reason === 'ValidationSucceeded' || condition.status === 'True',
    };
  } catch (error: any) {
    return {
      reason: 'ParseError',
      message: `Failed to parse bundle status: ${error.message}`,
      isValid: false,
    };
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { yaml } = body;

    if (!yaml || typeof yaml !== 'string') {
      return NextResponse.json(
        { error: 'Invalid YAML content' },
        { status: 400 }
      );
    }

    // Extract bundle name from YAML
    const bundleName = extractBundleName(yaml);
    if (!bundleName) {
      return NextResponse.json(
        { error: 'Could not extract bundle name from YAML' },
        { status: 400 }
      );
    }

    // Save YAML to temporary file
    const timestamp = Date.now();
    const fileName = `bundle-${timestamp}.yaml`;
    const filePath = path.join(process.cwd(), 'temp', fileName);

    // Ensure temp directory exists
    const tempDir = path.join(process.cwd(), 'temp');
    try {
      execSync(`mkdir -p "${tempDir}"`);
    } catch (error) {
      // Directory might already exist
    }

    // Write YAML to file
    writeFileSync(filePath, yaml, 'utf-8');

    // Set KUBECONFIG environment variable
    const kubeconfigPath = path.join(process.cwd(), 'kubeconfig.yaml');
    const env = { ...process.env, KUBECONFIG: kubeconfigPath };

    let applyOutput = '';
    try {
      // Run kubectl apply
      applyOutput = execSync(`kubectl apply -f "${filePath}"`, {
        encoding: 'utf-8',
        env,
        timeout: 30000, // 30 second timeout
      });
    } catch (error: any) {
      // kubectl apply failed
      return NextResponse.json(
        {
          success: false,
          error: 'kubectl apply failed',
          message: error.message || 'Unknown error',
          stderr: error.stderr?.toString() || '',
          stdout: error.stdout?.toString() || '',
          filePath,
        },
        { status: 400 }
      );
    }

    // Wait 5 seconds for bundle to be processed
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Get bundle status using JSON output
    let validationStatus;
    try {
      const jsonOutput = execSync(`kubectl get bundle ${bundleName} -o json`, {
        encoding: 'utf-8',
        env,
        timeout: 30000,
      });

      validationStatus = extractValidationStatus(jsonOutput);
    } catch (error: any) {
      // kubectl get bundle failed
      return NextResponse.json(
        {
          success: false,
          error: 'kubectl get bundle failed',
          message: 'Bundle was applied but status check failed',
          applyOutput: applyOutput.trim(),
          stderr: error.stderr?.toString() || '',
          stdout: error.stdout?.toString() || '',
        },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      message: validationStatus.isValid
        ? 'Bundle validation succeeded!'
        : 'Bundle validation failed',
      applyOutput: applyOutput.trim(),
      validationStatus,
      bundleName,
      filePath,
    });
  } catch (error: any) {
    console.error('Validation error:', error);
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
