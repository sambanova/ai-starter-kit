#!/usr/bin/env node

import { execSync } from 'child_process';
import { writeFileSync, readFileSync, existsSync } from 'fs';
import path from 'path';

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

interface PefConfig {
  ss: string;
  bs: string;
  latestVersion: string;
}

interface PefConfigs {
  [key: string]: PefConfig;
}

/**
 * Extract ss and bs values from PEF name
 * Example: "llama-3p1-70b-ss4096-bs1-sd9" -> ss: 4096, bs: 1
 */
function parsePefName(pefName: string): { ss: number; bs: number } | null {
  // Match pattern: ss followed by digits, bs followed by digits
  const ssMatch = pefName.match(/ss(\d+)/);
  const bsMatch = pefName.match(/bs(\d+)/);

  if (!ssMatch || !bsMatch) {
    console.warn(`Warning: Could not parse PEF name: ${pefName}`);
    return null;
  }

  return {
    ss: parseInt(ssMatch[1], 10),
    bs: parseInt(bsMatch[1], 10),
  };
}

/**
 * Convert ss value to "Xk" format
 */
function formatSsValue(ss: number): string {
  const ssInK = ss / 1024;
  return `${ssInK}k`;
}

/**
 * Get the latest version number from PEF description
 * Parses the Versions section and returns the highest version number
 */
function getLatestVersion(pefName: string, kubeconfigPath: string, namespace: string): string {
  try {
    const description = execSync(`kubectl -n ${namespace} describe pef ${pefName}`, {
      encoding: 'utf-8',
      env: { ...process.env, KUBECONFIG: kubeconfigPath },
    });

    // Find the Versions section and extract version numbers
    // Pattern matches lines like "  1:" or "  2:" in the Versions section
    const versionMatches = description.match(/Versions:\s*\n((?:\s+\d+:[\s\S]*?)+)/);

    if (!versionMatches) {
      console.warn(`Warning: No versions found for ${pefName}`);
      return '1'; // Default to version 1
    }

    // Extract all version numbers from the Versions section
    const versionsSection = versionMatches[1];
    const versionNumbers = versionsSection.match(/^\s+(\d+):/gm);

    if (!versionNumbers || versionNumbers.length === 0) {
      return '1';
    }

    // Extract numeric values and find the maximum
    const versions = versionNumbers.map((v) => parseInt(v.trim().replace(':', ''), 10));
    const latestVersion = Math.max(...versions);

    return latestVersion.toString();
  } catch (error) {
    console.warn(`Warning: Failed to get version for ${pefName}:`, error);
    return '1'; // Default to version 1 on error
  }
}

/**
 * Main function to generate PEF configs
 */
function generatePefConfigs() {
  console.log('Running kubectl get pefs...');

  // Read app-config.json to get current kubeconfig and namespace
  const configPath = path.join(__dirname, '..', 'app-config.json');
  if (!existsSync(configPath)) {
    console.error('Error: app-config.json not found. Please configure an environment first.');
    process.exit(1);
  }

  const configContent = readFileSync(configPath, 'utf-8');
  const config: AppConfig = JSON.parse(configContent);

  const currentEnv = config.currentKubeconfig;
  if (!currentEnv || !config.kubeconfigs[currentEnv]) {
    console.error('Error: No active environment configured. Please select an environment first.');
    process.exit(1);
  }

  const kubeconfigFile = config.kubeconfigs[currentEnv].file;
  const namespace = config.kubeconfigs[currentEnv].namespace || 'default';

  const kubeconfigPath = path.join(__dirname, '..', kubeconfigFile);
  if (!existsSync(kubeconfigPath)) {
    console.error(`Error: Kubeconfig file not found: ${kubeconfigFile}`);
    process.exit(1);
  }

  console.log(`Using environment: ${currentEnv}`);
  console.log(`Using namespace: ${namespace}`);
  console.log(`Using kubeconfig: ${kubeconfigFile}`);

  process.env.KUBECONFIG = kubeconfigPath;

  // Run kubectl command to get PEF names
  const kubectl = execSync(`kubectl -n ${namespace} get pefs --no-headers`, {
    encoding: 'utf-8',
    env: { ...process.env, KUBECONFIG: kubeconfigPath },
  });

  // Parse output to extract PEF names (first column)
  const pefNames = kubectl
    .trim()
    .split('\n')
    .map((line) => line.split(/\s+/)[0])
    .filter((name) => name);

  console.log(`Found ${pefNames.length} PEFs`);

  // Generate configs
  const configs: PefConfigs = {};

  for (let i = 0; i < pefNames.length; i++) {
    const pefName = pefNames[i];
    console.log(`Processing ${i + 1}/${pefNames.length}: ${pefName}`);

    const parsed = parsePefName(pefName);

    if (parsed) {
      const latestVersion = getLatestVersion(pefName, kubeconfigPath, namespace);

      configs[pefName] = {
        ss: formatSsValue(parsed.ss),
        bs: parsed.bs.toString(),
        latestVersion,
      };
    }
  }

  // Write to file
  const outputPath = path.join(__dirname, 'app', 'data', 'pef_configs.json');
  writeFileSync(outputPath, JSON.stringify(configs, null, 2), 'utf-8');

  console.log(`✓ Generated pef_configs.json with ${Object.keys(configs).length} entries`);
  console.log(`  Output: ${outputPath}`);
}

// Run the script
try {
  generatePefConfigs();
} catch (error) {
  console.error('Error generating PEF configs:', error);
  process.exit(1);
}
