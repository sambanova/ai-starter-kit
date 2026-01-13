export interface PefConfig {
  ss: string;
  bs: string;
  latestVersion: string;
}

export interface PefConfigs {
  [pefName: string]: PefConfig;
}

export interface PefMapping {
  [modelName: string]: string[];
}

export interface CheckpointMapping {
  [modelName: string]: {
    path: string;
    resource_name: string;
  };
}

export interface ConfigSelection {
  modelName: string;
  ss: string;
  bs: string;
  pefName: string;
}
