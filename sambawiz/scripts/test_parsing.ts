// Test file to debug kubectl describe bundle output parsing

// Get checkpoints directory from environment variable
const CHECKPOINTS_DIR = process.env.NEXT_PUBLIC_CHECKPOINTS_DIR || process.env.CHECKPOINTS_DIR || 'gs://your-bucket/path/to/checkpoints/';

const sampleOutput = `Name:         b-meta-llama-3-gpt-120b
Namespace:    default
Labels:       sambanova.ai/bundle-template=bt-meta-llama-3-gpt-120b
              sambanova.ai/bundle-template-version=3836915
Annotations:  kopf.zalando.org/last-handled-configuration:
                {"spec":{"checkpoints":{"GPT_OSS_120B_CKPT":{"source":"${CHECKPOINTS_DIR}gpt-oss-1...
API Version:  sambanova.ai/v1alpha1
Kind:         Bundle
Metadata:
  Creation Timestamp:  2026-01-07T01:27:53Z
  Generation:          1
  Resource Version:    3836933
  UID:                 8998aafd-fc3d-48dd-b115-7a192f10028e
Spec:
  Checkpoints:
    GPT_OSS_120B_CKPT:
      Source:        ${CHECKPOINTS_DIR}gpt-oss-120b-FP8-per-tensor-surgery
      Tool Support:  true
    META_LLAMA_3_2_3B_INSTRUCT_CKPT:
      Source:        ${CHECKPOINTS_DIR}meta-llama-Llama-3.2-3B-Instruct_untie
      Tool Support:  true
    META_LLAMA_3_3_70B_INSTRUCT_CKPT:
      Source:        ${CHECKPOINTS_DIR}Llama-3.3-70B-Instruct
      Tool Support:  true
  Models:
    gpt-oss-120b:
      Checkpoint:  GPT_OSS_120B_CKPT
      Template:    gpt-oss-120b
    meta-llama-3-2-3b-instruct:
      Checkpoint:  META_LLAMA_3_2_3B_INSTRUCT_CKPT
      Template:    meta-llama-3-2-3b-instruct
    meta-llama-3-3-70b-instruct:
      Checkpoint:  META_LLAMA_3_3_70B_INSTRUCT_CKPT
      Template:    meta-llama-3-3-70b-instruct
  Secret Names:
    sambanova-artifact-reader
  Template:  bt-meta-llama-3-gpt-120b
Status:
  Conditions:
    Last Transition Time:  2026-01-07T01:27:55.548271+00:00
    Message:               Validation Errors:
Bundle legalization failed for b-meta-llama-3-gpt-120b:
Bundle [b-meta-llama-3-gpt-120b]: invalid
errors for b-meta-llama-3-gpt-120b:
  - bundle uses 1968.8 MiB/pool more HBM memory than available


    Observed Generation:  1
    Reason:               ValidationFailed
    Status:               False
    Type:                 Valid
  Observed Generation:    1
Events:                   <none>`;

/**
 * Extract Status.Conditions section from kubectl describe output
 * OLD VERSION - doesn't work correctly
 */
function extractStatusConditionsOld(describeOutput: string): string {
  const lines = describeOutput.split('\n');
  const conditionsStartIndex = lines.findIndex(line => line.trim() === 'Conditions:');

  if (conditionsStartIndex === -1) {
    return 'No conditions found in bundle status';
  }

  // Find where the Conditions section ends (next top-level field or Events)
  let conditionsEndIndex = lines.length;
  for (let i = conditionsStartIndex + 1; i < lines.length; i++) {
    const line = lines[i];
    // Check if we've reached another top-level field (not indented)
    if (line.match(/^[A-Z]/) && !line.startsWith('  ')) {
      conditionsEndIndex = i;
      break;
    }
  }

  // Skip the "Conditions:" header line and return only the content beneath it
  return lines.slice(conditionsStartIndex + 1, conditionsEndIndex).join('\n').trim();
}

/**
 * Extract Status.Conditions section from kubectl describe output
 * NEW VERSION - fixed to handle multi-line fields properly
 *
 * The challenge: Conditions contains fields that have multi-line unindented content.
 * Solution: Only stop when we find a sibling field (same indentation + looks like a field name)
 */
function extractStatusConditionsNew(describeOutput: string): string {
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

console.log('=== Testing OLD version ===');
const oldResult = extractStatusConditionsOld(sampleOutput);
console.log(oldResult);
console.log(`Length: ${oldResult.length} characters`);

console.log('\n\n=== Testing NEW version ===');
const newResult = extractStatusConditionsNew(sampleOutput);
console.log(newResult);
console.log(`Length: ${newResult.length} characters`);

// Also test line-by-line to understand the structure
console.log('\n\n=== Line-by-line analysis around Conditions ===');
const lines = sampleOutput.split('\n');
const conditionsIdx = lines.findIndex(line => line.trim() === 'Conditions:');
const conditionsLine = lines[conditionsIdx];
const conditionsIndent = conditionsLine.length - conditionsLine.trimStart().length;
console.log(`Conditions line has ${conditionsIndent} spaces of indentation`);

for (let i = conditionsIdx; i < Math.min(conditionsIdx + 20, lines.length); i++) {
  const line = lines[i];
  const lineIndent = line.length - line.trimStart().length;
  const hasColon = line.includes(':');
  const shouldEnd = lineIndent <= conditionsIndent && hasColon && line.trim() !== '';
  console.log(`Line ${i}: indent=${lineIndent} hasColon=${hasColon} shouldEnd=${shouldEnd} "${line}"`);
}
