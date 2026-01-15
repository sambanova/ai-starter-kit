import type { ConfigSelection, CheckpointMapping, PefConfigs } from '../types/bundle';

interface ModelExpertConfig {
  batch_size: string;
  ckpt_sharing_uuid: string;
  num_tokens_at_a_time: number;
  pef: string;
  spec_decoding?: {
    draft_expert: string;
    draft_model: string;
  };
}

interface ModelExperts {
  [ss: string]: {
    configs: ModelExpertConfig[];
  };
}

interface BundleTemplateModels {
  [modelName: string]: {
    experts: ModelExperts;
  };
}

/**
 * Generate checkpoint name from model name
 */
export function generateCheckpointName(modelName: string): string {
  return modelName
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '') + '_CKPT';
}

/**
 * Generate complete bundle YAML from selected configurations
 */
export function generateBundleYaml(
  selectedConfigs: ConfigSelection[],
  checkpointMapping: CheckpointMapping,
  pefConfigs: PefConfigs,
  bundleName: string,
  checkpointsDir: string = '',
  draftModels: { [modelName: string]: string } = {}
): string {
  // Group configs by model
  const modelConfigs: { [modelName: string]: ConfigSelection[] } = {};
  selectedConfigs.forEach(config => {
    if (!modelConfigs[config.modelName]) {
      modelConfigs[config.modelName] = [];
    }
    modelConfigs[config.modelName].push(config);
  });

  // Build BundleTemplate spec.models
  const templateModels: BundleTemplateModels = {};
  let uuidCounter = 1; // Counter for ckpt_sharing_uuid across all models

  Object.entries(modelConfigs).forEach(([modelName, configs]) => {
    const experts: ModelExperts = {};

    // Check if this model has a draft model assigned
    const draftModel = draftModels[modelName];
    const hasDraftModel = draftModel && draftModel !== 'skip';

    // Group by SS and assign UUID per expert
    configs.forEach(config => {
      if (!experts[config.ss]) {
        experts[config.ss] = { configs: [] };
      }

      const version = pefConfigs[config.pefName]?.latestVersion || '1';

      // Check if this config will have spec_decoding (i.e., is a target model with matching draft config)
      let hasSpecDecoding = false;
      if (hasDraftModel) {
        const draftModelHasMatchingConfig = selectedConfigs.some(
          (sc) => sc.modelName === draftModel && sc.ss === config.ss && sc.bs === config.bs
        );
        hasSpecDecoding = draftModelHasMatchingConfig;
      }

      const expertConfig: ModelExpertConfig = {
        batch_size: config.bs,
        ckpt_sharing_uuid: '', // Will be set below
        num_tokens_at_a_time: hasSpecDecoding ? 1 : 20, // Target models use 1, others use 20
        pef: `${config.pefName}:${version}`
      };

      // Add spec_decoding if this model has a draft model and the draft model has a matching config
      if (hasSpecDecoding) {
        expertConfig.spec_decoding = {
          draft_expert: config.ss,
          draft_model: draftModel
        };
      }

      experts[config.ss].configs.push(expertConfig);
    });

    // Assign ckpt_sharing_uuid to each expert (all configs in an expert get the same ID)
    Object.values(experts).forEach(expert => {
      const currentUuid = `id${uuidCounter}`;
      expert.configs.forEach(config => {
        config.ckpt_sharing_uuid = currentUuid;
      });
      uuidCounter++;
    });

    templateModels[modelName] = { experts };
  });

  // Build Bundle spec.checkpoints
  const checkpoints: { [key: string]: { source: string; toolSupport: boolean } } = {};
  Object.keys(modelConfigs).forEach(modelName => {
    const checkpointName = generateCheckpointName(modelName);
    const checkpointData = checkpointMapping[modelName];
    const checkpointPath = checkpointData?.path || '';
    const fullCheckpointPath = checkpointsDir ? `${checkpointsDir}${checkpointPath}` : checkpointPath;
    checkpoints[checkpointName] = {
      source: fullCheckpointPath,
      toolSupport: true
    };
  });

  // Build Bundle spec.models
  const bundleModels: { [key: string]: { checkpoint: string; template: string } } = {};
  Object.keys(modelConfigs).forEach(modelName => {
    const checkpointName = generateCheckpointName(modelName);
    bundleModels[modelName] = {
      checkpoint: checkpointName,
      template: modelName
    };
  });

  // Generate YAML strings
  const bundleTemplateName = `bt-${bundleName}`;
  const bundleManifestName = `b-${bundleName}`;

  const bundleTemplateYaml = `apiVersion: sambanova.ai/v1alpha1
kind: BundleTemplate
metadata:
  name: ${bundleTemplateName}
spec:
  models:
${Object.entries(templateModels).map(([modelName, model]) => {
  return `    ${modelName}:
      experts:
${Object.entries(model.experts).map(([ss, expert]) => {
  return `        ${ss}:
          configs:
${expert.configs.map(config => {
  let configStr = `          - batch_size: ${config.batch_size}
            ckpt_sharing_uuid: ${config.ckpt_sharing_uuid}
            num_tokens_at_a_time: ${config.num_tokens_at_a_time}
            pef: ${config.pef}`;
  if (config.spec_decoding) {
    configStr += `
            spec_decoding:
              draft_expert: ${config.spec_decoding.draft_expert}
              draft_model: ${config.spec_decoding.draft_model}`;
  }
  return configStr;
}).join('\n')}`;
}).join('\n')}`;
}).join('\n')}
  owner: no-reply@sambanova.ai
  secretNames:
  - sambanova-artifact-reader
  usePefCRs: true`;

  const bundleYaml = `apiVersion: sambanova.ai/v1alpha1
kind: Bundle
metadata:
  name: ${bundleManifestName}
spec:
  checkpoints:
${Object.entries(checkpoints).map(([name, checkpoint]) => {
  return `    ${name}:
      source: ${checkpoint.source}
      toolSupport: ${checkpoint.toolSupport}`;
}).join('\n')}
  models:
${Object.entries(bundleModels).map(([modelName, model]) => {
  return `    ${modelName}:
      checkpoint: ${model.checkpoint}
      template: ${model.template}`;
}).join('\n')}
  secretNames:
  - sambanova-artifact-reader
  template: ${bundleTemplateName}`;

  return `${bundleTemplateYaml}\n---\n${bundleYaml}\n`;
}
