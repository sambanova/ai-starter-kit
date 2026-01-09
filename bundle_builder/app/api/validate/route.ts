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
 * Extract Status.Conditions section from kubectl describe output
 *
 * The challenge: Conditions contains fields that have multi-line unindented content.
 * Solution: Only stop when we find a sibling field (same indentation + looks like a field name)
 */
function extractStatusConditions(describeOutput: string): string {
  const lines = describeOutput.split('\n');
  const conditionsStartIndex = lines.findIndex(line => line.trim() === 'Conditions:');

  if (conditionsStartIndex === -1) {
    return 'No conditions found in bundle status';
  }

  // Determine the indentation level of the Conditions line
  const conditionsLine = lines[conditionsStartIndex];
  const conditionsIndent = conditionsLine.length - conditionsLine.trimStart().length;

  // Find where the Conditions section ends
  // Look for the next field at the SAME indentation level (sibling field)
  // Fields at this level look like "  FieldName:" with exactly that indentation
  let conditionsEndIndex = lines.length;
  for (let i = conditionsStartIndex + 1; i < lines.length; i++) {
    const line = lines[i];

    // Skip empty lines
    if (line.trim() === '') {
      continue;
    }

    // Calculate indentation
    const lineIndent = line.length - line.trimStart().length;

    // Check if this is a sibling field at the Status level
    // It must have:
    // 1. Exactly the same indentation as "Conditions:"
    // 2. Start with spaces followed by a capital letter or number
    // 3. End with a colon (be a field name)
    if (lineIndent === conditionsIndent) {
      const trimmed = line.trim();
      // Check if it looks like a field name (starts with capital/digit, ends with colon)
      if (trimmed.match(/^[A-Z0-9]/) && trimmed.endsWith(':')) {
        conditionsEndIndex = i;
        break;
      }
    }
  }

  // Skip the "Conditions:" header line and return only the content beneath it
  return lines.slice(conditionsStartIndex + 1, conditionsEndIndex).join('\n').trim();
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

    // Get bundle status
    let statusConditions = '';
    try {
      const describeOutput = execSync(`kubectl describe bundle ${bundleName}`, {
        encoding: 'utf-8',
        env,
        timeout: 30000,
      });

      statusConditions = extractStatusConditions(describeOutput);
    } catch (error: any) {
      // kubectl describe failed
      return NextResponse.json(
        {
          success: false,
          error: 'kubectl describe bundle failed',
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
      message: 'Bundle validated and applied successfully',
      applyOutput: applyOutput.trim(),
      statusConditions,
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
