import { NextRequest, NextResponse } from 'next/server';
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

export async function POST(request: NextRequest) {
  try {
    const { message, model, messages } = await request.json();

    if ((!message && !messages) || !model) {
      return NextResponse.json({
        success: false,
        error: 'Message/messages and model are required',
      }, { status: 400 });
    }

    // Read app-config.json to get API credentials
    const configPath = path.join(process.cwd(), 'app-config.json');

    if (!fs.existsSync(configPath)) {
      return NextResponse.json({
        success: false,
        error: 'app-config.json not found',
      }, { status: 500 });
    }

    const configContent = fs.readFileSync(configPath, 'utf-8');
    const config: AppConfig = JSON.parse(configContent);

    const currentEnvironment = config.currentKubeconfig;

    if (!currentEnvironment || !config.kubeconfigs[currentEnvironment]) {
      return NextResponse.json({
        success: false,
        error: `Current environment ${currentEnvironment} not found in app-config.json`,
      }, { status: 500 });
    }

    const environmentConfig = config.kubeconfigs[currentEnvironment];
    const apiKey = environmentConfig.apiKey;
    const apiDomain = environmentConfig.apiDomain;

    // Validate API key and domain
    if (!apiKey) {
      return NextResponse.json({
        success: false,
        error: `API Key not found in app-config.json for environment ${currentEnvironment}`,
      }, { status: 500 });
    }

    if (!apiDomain) {
      return NextResponse.json({
        success: false,
        error: `API Domain not found in app-config.json for environment ${currentEnvironment}`,
      }, { status: 500 });
    }

    // Ensure apiDomain ends with a slash
    const normalizedApiDomain = apiDomain.endsWith('/') ? apiDomain : `${apiDomain}/`;

    // Build messages array
    let messagesList;
    if (messages) {
      // Use provided message history
      messagesList = messages;
    } else {
      // Single message (backward compatibility)
      messagesList = [
        {
          role: 'system',
          content: 'You are a helpful assistant',
        },
        {
          role: 'user',
          content: message,
        },
      ];
    }

    // Make the API call to the chat completions endpoint
    const apiUrl = `${normalizedApiDomain}v1/chat/completions`;

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        stream: false,
        model: model,
        messages: messagesList,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', errorText);
      return NextResponse.json({
        success: false,
        error: `API request failed: ${response.status} ${response.statusText}`,
      }, { status: response.status });
    }

    const data = await response.json();

    // Extract the assistant's response
    const assistantMessage = data.choices?.[0]?.message?.content;

    if (!assistantMessage) {
      return NextResponse.json({
        success: false,
        error: 'No response from the model',
      }, { status: 500 });
    }

    // Extract usage metrics
    const usage = data.usage;
    const metrics = {
      tokensPerSecond: usage?.completion_tokens_after_first_per_sec || null,
      totalLatency: usage?.total_latency || null,
      timeToFirstToken: usage?.time_to_first_token || null,
    };

    return NextResponse.json({
      success: true,
      content: assistantMessage,
      metrics,
    });

  } catch (error: any) {
    console.error('Error in chat API:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to process chat request',
    }, { status: 500 });
  }
}
