/**
 * Edge GUI Agent Demo — Control Panel App Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  isRunning: false,
  executionSteps: [],
  workflows: [],
};

// ---------------------------------------------------------------------------
// LLM Configuration — replace with your own API details
// ---------------------------------------------------------------------------

const LLM_CONFIG = {
  enabled: true,
  endpoint: 'https://www.dmxapi.cn/v1/chat/completions',
  apiKey: 'REDACTED_DMXAPI_KEY',
  model: 'deepseek-v4-pro-guan',
};

// ---------------------------------------------------------------------------
// VLM Configuration — pending; URL and key to be provided
// ---------------------------------------------------------------------------

const VLM_CONFIG = {
  enabled: true,
  endpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
  apiKey: 'REDACTED_DASHSCOPE_KEY',
  model: 'qwen-vl-max',
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function initApp() {
  bindUI();
  bindEvents();
  await loadWorkflows();

  logMessage('info', 'Edge GUI Agent Demo initialized');
  logMessage('info', 'Enter a prompt and click "Run Agent" or select a sample workflow');
}

// ---------------------------------------------------------------------------
// UI Bindings
// ---------------------------------------------------------------------------

function bindUI() {
  state.$url = document.getElementById('target-url');
  state.$prompt = document.getElementById('agent-prompt');
  state.$btnRun = document.getElementById('btn-run');
  state.$btnStop = document.getElementById('btn-stop');
  state.$btnReload = document.getElementById('btn-reload-sandbox');
  state.$btnNavigate = document.getElementById('btn-navigate');
  state.$statusBar = document.getElementById('status-bar');
  state.$logOutput = document.getElementById('log-output');
  state.$workflowSelect = document.getElementById('workflow-select');
  state.$btnLoadWorkflow = document.getElementById('btn-load-workflow');
  state.$btnClearLog = document.getElementById('btn-clear-log');
  state.$modelSelect = document.getElementById('model-select');
}

function bindEvents() {
  // Navigate
  state.$btnNavigate.addEventListener('click', handleNavigate);
  state.$url.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleNavigate();
  });

  // Agent control
  state.$btnRun.addEventListener('click', handleRunAgent);
  state.$btnStop.addEventListener('click', handleStopAgent);
  state.$btnReload.addEventListener('click', handleReloadSandbox);

  // Workflows
  state.$btnLoadWorkflow.addEventListener('click', handleLoadWorkflow);

  // Clear log
  state.$btnClearLog.addEventListener('click', () => {
    state.$logOutput.innerHTML = '';
  });

  // Quick actions
  document.querySelectorAll('.qa-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const action = btn.dataset.action;
      if (action === 'getState') {
        await handleGetState();
      } else if (action === 'screenshot') {
        await handleScreenshot();
      }
    });
  });

  // Sandbox events
  if (window.electronAPI) {
    window.electronAPI.onSandboxLog((data) => {
      logMessage('sandbox', data.message);
    });

    window.electronAPI.onSandboxNavigated((url) => {
      state.$url.value = url;
      logMessage('info', `Sandbox navigated to: ${url}`);
    });
  }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async function handleNavigate() {
  const url = state.$url.value.trim();
  if (!url) return;

  updateStatus('navigating', `Navigating to ${url}...`);

  try {
    let finalUrl = url;
    if (!/^https?:\/\//i.test(finalUrl)) {
      finalUrl = 'https://' + finalUrl;
    }

    const result = await window.electronAPI?.navigate(finalUrl);
    if (result?.success) {
      updateStatus('idle', 'Ready');
      logMessage('success', `Navigated to ${finalUrl}`);
    } else {
      updateStatus('error', 'Navigation failed');
      logMessage('error', `Navigation error: ${result?.error || 'Unknown'}`);
    }
  } catch (err) {
    updateStatus('error', err.message);
    logMessage('error', err.message);
  }
}

async function handleRunAgent() {
  const prompt = state.$prompt.value.trim();
  if (!prompt) {
    logMessage('warn', 'Please enter a prompt first');
    return;
  }

  state.isRunning = true;
  state.executionSteps = [];
  state.$btnRun.disabled = true;
  state.$btnStop.disabled = false;
  updateStatus('running', 'Agent executing...');
  logMessage('info', `Starting agent execution: "${prompt}"`);

  try {
    await executeWorkflow(prompt);
  } catch (err) {
    logMessage('error', `Agent execution failed: ${err.message}`);
  } finally {
    state.isRunning = false;
    state.$btnRun.disabled = false;
    state.$btnStop.disabled = true;
    updateStatus('idle', `Completed — ${state.executionSteps.length} steps`);
  }
}

function handleStopAgent() {
  state.isRunning = false;
  state.$btnRun.disabled = false;
  state.$btnStop.disabled = true;
  updateStatus('idle', 'Stopped by user');
  logMessage('warn', 'Agent execution stopped by user');
}

async function handleReloadSandbox() {
  try {
    await window.electronAPI?.reloadSandbox();
    updateStatus('idle', 'Page reloaded');
    logMessage('info', 'Sandbox page reloaded');
  } catch (err) {
    logMessage('error', `Reload failed: ${err.message}`);
  }
}

async function handleGetState() {
  try {
    const result = await window.electronAPI?.execute({ type: 'getState' });
    if (result?.success) {
      const state_data = result.result;
      logMessage('success', `Page state: ${state_data.page.url} — ${state_data.interactiveElements.length} interactive elements`);
    } else {
      logMessage('error', `getState failed: ${result?.error}`);
    }
  } catch (err) {
    logMessage('error', err.message);
  }
}

async function handleScreenshot() {
  try {
    updateStatus('screenshot', 'Capturing screenshot...');
    const result = await window.electronAPI?.screenshot();
    if (result?.success) {
      if (result.filePath) {
        logMessage('success', `Screenshot saved to: ${result.filePath}`);
      } else {
        logMessage('success', 'Screenshot captured (base64)');
      }
      updateStatus('idle', 'Screenshot saved');
    } else {
      logMessage('error', `Screenshot failed: ${result?.error}`);
      updateStatus('error', 'Screenshot failed');
    }
  } catch (err) {
    logMessage('error', err.message);
    updateStatus('error', err.message);
  }
}

async function handleLoadWorkflow() {
  const selectedId = state.$workflowSelect.value;
  if (!selectedId) return;

  const workflow = state.workflows.find((w) => w.id === selectedId);
  if (!workflow) return;

  state.$prompt.value = workflow.prompt;
  logMessage('info', `Loaded workflow: ${workflow.name}`);
}

// ---------------------------------------------------------------------------
// Agent Execution Engine
// ---------------------------------------------------------------------------

async function executeWorkflow(prompt) {
  // Step 1: Get page state
  logMessage('info', 'Step 1: Getting page state...');
  const stateResult = await window.electronAPI?.execute({ type: 'getState' });

  if (!state.isRunning) return;
  if (!stateResult?.success) {
    logMessage('error', 'Failed to get page state');
    return;
  }

  const pageState = stateResult.result;
  const elements = pageState.interactiveElements || [];
  logMessage('success', `Found ${elements.length} interactive elements`);

  // Step 2: Plan actions — route to selected model/parser
  logMessage('info', 'Step 2: Planning actions...');
  await sleep(300);

  let actions;
  const modelChoice = state.$modelSelect?.value || 'text-llm';

  if (modelChoice === 'text-llm' && LLM_CONFIG.enabled) {
    logMessage('info', 'Asking LLM to plan actions...');
    actions = await callLLM(prompt, elements);
    if (actions && actions.length > 0) {
      logMessage('success', `LLM planned ${actions.length} actions`);
    } else {
      logMessage('warn', 'LLM returned no actions, falling back to regex parser');
    }
  } else if (modelChoice === 'vlm' && VLM_CONFIG.enabled) {
    logMessage('info', 'Asking VLM to plan actions (with screenshot)...');
    actions = await callVLM(prompt, elements);
    if (actions && actions.length > 0) {
      logMessage('success', `VLM planned ${actions.length} actions`);
    } else {
      logMessage('warn', 'VLM returned no actions, falling back to regex parser');
    }
  }

  if (!actions || actions.length === 0) {
    actions = parsePromptToActions(prompt, elements);
    logMessage('info', `Regex parser generated ${actions.length} actions`);
  }

  if (actions.length === 0) {
    if (actions._displayOnly) {
      dumpElementsToLog(elements);
      logMessage('success', `Done — ${elements.length} interactive elements listed above`);
    } else {
      logMessage('warn', 'No matching actions found for this prompt. Try a different prompt.');
    }
    return;
  }

  // Step 3: Execute actions
  for (let i = 0; i < actions.length; i++) {
    if (!state.isRunning) break;

    const action = actions[i];
    logMessage('info', `Step ${i + 3}: Executing ${action.type} on "${action.description || action.selector || `(${action.x}, ${action.y})`}"`);
    state.executionSteps.push(action);

    await sleep(800);

    const result = await window.electronAPI?.execute(action);
    if (result?.success) {
      logMessage('success', `✓ ${action.type} completed — ${JSON.stringify(result.result)}`);
    } else {
      logMessage('error', `✗ ${action.type} failed — ${result?.error}`);
    }

    await sleep(300);
  }

  // Step 4: If this was a search workflow, re-query page state and click first result
  if (actions._needsPostSearchClick && state.isRunning) {
    logMessage('info', 'Step 4: Re-querying page state for search results...');
    await sleep(1500);

    const freshState = await window.electronAPI?.execute({ type: 'getState' });
    if (!state.isRunning) return;
    if (!freshState?.success) {
      logMessage('warn', 'Failed to re-query page state after search');
      return;
    }

    const freshElements = freshState.result.interactiveElements || [];
    logMessage('success', `Found ${freshElements.length} elements on results page`);

    // Find first meaningful result link:
    // - text length ≥ 15 chars (weed out nav tabs like "Images"/"Videos"/"News")
    // - y position ≥ 120 (skip top header/nav bar)
    // - skip common nav/search-tab labels
    const navLabels = ['images', 'videos', 'news', 'maps', 'shopping', 'video', 'image',
      '图片', '视频', '新闻', '地图', '购物', 'sign in', 'sign up', 'log in', 'login',
      'settings', 'help', 'feedback', 'privacy', 'terms', 'cookies', 'skip to', 'account'];
    const isResultLink = (el) => {
      if (el.tag !== 'a') return false;
      const txt = (el.text || '').trim().toLowerCase();
      if (txt.length < 15) return false;
      if (navLabels.some(l => txt === l || txt.startsWith(l + ' '))) return false;
      if (el.position && el.position.y < 120) return false;
      return true;
    };
    const firstResultLink = freshElements.find(isResultLink);
    if (firstResultLink) {
      logMessage('info', `Clicking first result: ${firstResultLink.text.slice(0, 40)}`);
      await sleep(500);
      const clickResult = await window.electronAPI?.execute({
        type: 'click',
        x: firstResultLink.position.x,
        y: firstResultLink.position.y,
      });
      if (clickResult?.success) {
        logMessage('success', `✓ clicked first result — ${JSON.stringify(clickResult.result)}`);
      } else {
        logMessage('error', `✗ click failed — ${clickResult?.error}`);
      }
    } else {
      logMessage('warn', 'No link found on results page');
    }
  }
}

// ---------------------------------------------------------------------------
// LLM-Powered Action Planner — sends prompt + page state to LLM, returns actions
// ---------------------------------------------------------------------------

const LLM_SYSTEM_PROMPT = `You are a browser automation agent. Your job is to translate a user's natural-language instruction into a sequence of UI actions on a web page.

You will receive:
1. The user's instruction (prompt)
2. A JSON array of interactive elements on the current page

Each element has: tag, type, text, value, name, id, placeholder, ariaLabel, autocomplete, href, position {x, y}, selector.

Available action types:
- click: { type: "click", x: number, y: number, description: "what you are clicking" }
- type: { type: "type", text: "string to type", selector: "css selector", clear: true|false, description: "what you are typing" }
- press: { type: "press", key: "Enter"|"Tab"|"Escape"|..., description: "what key you are pressing" }
- scroll: { type: "scroll", direction: "up"|"down", amount: number(pixels), description: "scrolling direction" }
- wait: { type: "wait", ms: number(milliseconds), description: "reason to wait" }
- getState: { type: "getState", description: "capture page state" }

Rules:
1. Find the most relevant element(s) matching the user's instruction. Match by text, name, id, placeholder, type, ariaLabel, autocomplete — NOT by vague guesses.
2. For text input, ALWAYS click the input first (focus it), then type.
3. After submitting a form or triggering navigation, add a wait action to let the page load.
4. For "show me" / "list" / "display" / "capture state" prompts, return ONLY [{ "type": "getState", "description": "capture page state" }].
5. For "scroll" prompts, return scroll action(s) followed by getState.
6. If no matching element exists, return [{ "type": "getState", "description": "no matching elements — capture page state" }].

Output: Return ONLY a valid JSON array of action objects. No markdown, no explanation, no code fences.`;

async function callLLM(prompt, elements) {
  if (!LLM_CONFIG.enabled) return null;

  // Build a compact elements summary for the LLM
  const elementsSummary = elements.map((el, i) => ({
    idx: i,
    tag: el.tag,
    type: el.type || '',
    text: (el.text || '').slice(0, 80),
    value: el.value || '',
    name: el.name || '',
    id: el.id || '',
    placeholder: el.placeholder || '',
    ariaLabel: el.ariaLabel || '',
    autocomplete: el.autocomplete || '',
    href: (el.href || '').slice(0, 100),
    x: Math.round(el.position?.x || 0),
    y: Math.round(el.position?.y || 0),
    selector: el.selector || '',
  }));

  const userMessage = JSON.stringify({
    prompt: prompt,
    elements: elementsSummary,
  });

  logMessage('info', `Calling LLM (${elementsSummary.length} elements)...`);

  try {
    const result = await window.electronAPI?.llmCall({
      endpoint: LLM_CONFIG.endpoint,
      apiKey: LLM_CONFIG.apiKey,
      model: LLM_CONFIG.model,
      body: {
        model: LLM_CONFIG.model,
        messages: [
          { role: 'system', content: LLM_SYSTEM_PROMPT },
          { role: 'user', content: userMessage },
        ],
        temperature: 0.1,
        max_tokens: 1024,
      },
    });

    if (!result?.success) {
      logMessage('error', `LLM API error: ${result?.error || 'unknown'}`);
      return null;
    }

    let content = result.data || '';
    try {
      const parsed = JSON.parse(content);
      content = parsed.choices?.[0]?.message?.content || content;
    } catch {}

    // Strip markdown fences if present
    content = content.trim();
    if (content.startsWith('```')) {
      content = content.replace(/^```(?:json)?\s*\n?/, '').replace(/\n?```\s*$/, '');
    }

    const actions = JSON.parse(content);
    if (!Array.isArray(actions)) {
      logMessage('error', 'LLM returned non-array actions');
      return null;
    }

    logMessage('success', `LLM returned ${actions.length} actions`);
    return actions;
  } catch (err) {
    logMessage('error', `LLM call failed: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// VLM-Powered Action Planner — sends screenshot + prompt + page state to VLM
// ---------------------------------------------------------------------------

const VLM_SYSTEM_PROMPT = `You are a browser automation agent with vision capabilities. Your job is to translate a user's natural-language instruction into a sequence of UI actions on a web page.

You will receive:
1. The user's instruction (prompt)
2. A JSON array of interactive elements on the current page
3. A screenshot of the current page

Each element has: tag, type, text, value, name, id, placeholder, ariaLabel, autocomplete, href, position {x, y}, selector.

Available action types:
- click: { type: "click", x: number, y: number, description: "what you are clicking" }
- type: { type: "type", text: "string to type", selector: "css selector", clear: true|false, description: "what you are typing" }
- press: { type: "press", key: "Enter"|"Tab"|"Escape"|..., description: "what key you are pressing" }
- scroll: { type: "scroll", direction: "up"|"down", amount: number(pixels), description: "scrolling direction" }
- wait: { type: "wait", ms: number(milliseconds), description: "reason to wait" }
- getState: { type: "getState", description: "capture page state" }

Rules:
1. Use the screenshot to visually verify element positions and layout.
2. Find the most relevant element(s) matching the user's instruction. Match by text, name, id, placeholder, type, ariaLabel, autocomplete — cross-reference with the screenshot.
3. For text input, ALWAYS click the input first (focus it), then type.
4. After submitting a form or triggering navigation, add a wait action to let the page load.
5. For "show me" / "list" / "display" / "capture state" prompts, return ONLY [{ "type": "getState", "description": "capture page state" }].
6. For "scroll" prompts, return scroll action(s) followed by getState.
7. If no matching element exists, return [{ "type": "getState", "description": "no matching elements — capture page state" }].

Output: Return ONLY a valid JSON array of action objects. No markdown, no explanation, no code fences.`;

async function callVLM(prompt, elements) {
  if (!VLM_CONFIG.enabled) return null;

  // Step 1: Capture screenshot
  logMessage('info', 'Capturing screenshot for VLM...');
  let imageBase64;
  try {
    const screenshotResult = await window.electronAPI?.screenshot();
    if (!screenshotResult?.success || !screenshotResult.image) {
      logMessage('error', 'Failed to capture screenshot for VLM');
      return null;
    }
    imageBase64 = screenshotResult.image;
  } catch (err) {
    logMessage('error', `Screenshot capture error: ${err.message}`);
    return null;
  }

  // Step 2: Build elements summary (same format as LLM)
  const elementsSummary = elements.map((el, i) => ({
    idx: i,
    tag: el.tag,
    type: el.type || '',
    text: (el.text || '').slice(0, 80),
    value: el.value || '',
    name: el.name || '',
    id: el.id || '',
    placeholder: el.placeholder || '',
    ariaLabel: el.ariaLabel || '',
    autocomplete: el.autocomplete || '',
    href: (el.href || '').slice(0, 100),
    x: Math.round(el.position?.x || 0),
    y: Math.round(el.position?.y || 0),
    selector: el.selector || '',
  }));

  const textPart = JSON.stringify({ prompt, elements: elementsSummary });

  // Step 3: Send to VLM — vision model expects multimodal content array
  logMessage('info', `Calling VLM with screenshot + ${elementsSummary.length} elements...`);

  try {
    const result = await window.electronAPI?.llmCall({
      endpoint: VLM_CONFIG.endpoint,
      apiKey: VLM_CONFIG.apiKey,
      model: VLM_CONFIG.model,
      body: {
        model: VLM_CONFIG.model,
        messages: [
          { role: 'system', content: VLM_SYSTEM_PROMPT },
          {
            role: 'user',
            content: [
              { type: 'text', text: textPart },
              { type: 'image_url', image_url: { url: imageBase64 } },
            ],
          },
        ],
        temperature: 0.1,
        max_tokens: 1024,
      },
    });

    if (!result?.success) {
      logMessage('error', `VLM API error: ${result?.error || 'unknown'}`);
      return null;
    }

    let content = result.data || '';
    try {
      const parsed = JSON.parse(content);
      content = parsed.choices?.[0]?.message?.content || content;
    } catch {}

    content = content.trim();
    if (content.startsWith('```')) {
      content = content.replace(/^```(?:json)?\s*\n?/, '').replace(/\n?```\s*$/, '');
    }

    const actions = JSON.parse(content);
    if (!Array.isArray(actions)) {
      logMessage('error', 'VLM returned non-array actions');
      return null;
    }

    logMessage('success', `VLM returned ${actions.length} actions`);
    return actions;
  } catch (err) {
    logMessage('error', `VLM call failed: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Smart Prompt Parser — finds real form elements on any website
// ---------------------------------------------------------------------------

function parsePromptToActions(prompt, elements) {
  const actions = [];
  const promptLower = prompt.toLowerCase();

  console.log('[DEBUG] Parsing prompt:', prompt);
  console.log('[DEBUG] Elements on page:', elements.length);

  // Dump all inputs / buttons to console for debugging
  const inputs = elements.filter(e => e.tag === 'input');
  const buttons = elements.filter(e => e.tag === 'button' || (e.tag === 'input' && (e.type === 'submit' || e.type === 'button')));
  console.log('[DEBUG] Inputs:', inputs.map(e => ({ tag: e.tag, type: e.type, name: e.name, id: e.id, placeholder: e.placeholder, ariaLabel: e.ariaLabel })));
  console.log('[DEBUG] Buttons:', buttons.map(e => ({ tag: e.tag, type: e.type, text: e.text, value: e.value, id: e.id })));

  // ========================================================================
  // 1. SEARCH WORKFLOW: "Search for 'XXX' and click the first result"
  //    Returns only the search-submit phase. Post-navigation click is handled
  //    by executeWorkflow() which re-queries the page after search.
  // ========================================================================
  if (promptLower.includes('search for') || promptLower.includes('search')) {
    const searchMatch = prompt.match(/search\s+(?:for\s+)?["']?([^"'\n]+)["']?/i);
    const query = searchMatch ? searchMatch[1].trim() : 'AI news';
    console.log('[DEBUG] Search query:', query);

    // --- Find search input ---
    const searchKeywords = ['search', 'q', 'query', 'keyword', 'find', 'buscar', 'suche', 'recherche'];
    const isSearchInput = (el) => {
      if (el.tag !== 'input' && el.tag !== 'textarea') return false;
      if (el.type === 'search') return true;
      const text = [
        (el.name || '').toLowerCase(),
        (el.id || '').toLowerCase(),
        (el.placeholder || '').toLowerCase(),
        (el.ariaLabel || '').toLowerCase(),
      ].join(' ');
      return searchKeywords.some(k => text.includes(k));
    };

    let searchBox = elements.find(isSearchInput);
    if (!searchBox) {
      searchBox = elements.find(e =>
        (e.tag === 'input' && (e.type === 'text' || !e.type)) || e.tag === 'textarea'
      );
    }
    console.log('[DEBUG] Search box:', !!searchBox, searchBox);

    // --- Find search button ---
    const searchBtnKeywords = ['search', 'go', 'submit', 'find', 'buscar', 'suche', 'recherche', '搜'];
    const isSearchButton = (el) => {
      if (el.tag !== 'button' && !(el.tag === 'input' && (el.type === 'submit' || el.type === 'button'))) return false;
      const text = [
        (el.text || '').toLowerCase(),
        (el.value || '').toLowerCase(),
        (el.ariaLabel || '').toLowerCase(),
        (el.name || '').toLowerCase(),
        (el.id || '').toLowerCase(),
      ].join(' ');
      return searchBtnKeywords.some(k => text.includes(k));
    };
    const searchButton = elements.find(isSearchButton);
    console.log('[DEBUG] Search button:', !!searchButton, searchButton);

    if (searchBox) {
      actions.push({ type: 'click', x: searchBox.position.x, y: searchBox.position.y, description: 'Focus search box' });
      actions.push({ type: 'type', text: query, selector: searchBox.selector, clear: true, description: `Type "${query}"` });
      actions.push({ type: 'wait', ms: 300, description: 'Wait for input' });
    }

    // Try clicking the search button first (more reliable than Enter)
    if (searchButton) {
      actions.push({ type: 'click', x: searchButton.position.x, y: searchButton.position.y, description: 'Click search button' });
    } else {
      actions.push({ type: 'press', key: 'Enter', description: 'Press Enter to search' });
    }

    // Wait for search results to load
    actions.push({ type: 'wait', ms: 3000, description: 'Wait for results' });

    // Only click a result if the user explicitly asked for it
    const wantsClick = /\b(?:click|open|navigate to|go to|follow|visit)\s+(?:the\s+)?(?:first|top|1st|\d+..)?\s*(?:result|link|search result)/i.test(promptLower);
    if (wantsClick) {
      actions._needsPostSearchClick = true;
    }
    return actions;
  }

  // ========================================================================
  // 2. LOGIN WORKFLOW: "Log in with username 'XXX' and password 'XXX'"
  // ========================================================================
  if (promptLower.includes('log in') || promptLower.includes('login') ||
      promptLower.includes('sign in') || promptLower.includes('signin')) {
    const loginMatch = prompt.match(/(?:log\s*in|login|sign\s*in|signin)\s+(?:with\s+)?(?:username\s+["']?([^"'\n\s]+)["']?\s*)?(?:and\s+)?(?:password\s+["']?([^"'\n\s]+)["']?)?/i);
    const username = loginMatch?.[1] || 'demo';
    const password = loginMatch?.[2] || 'demo123';
    console.log('[DEBUG] Login credentials:', { username, password });

    // --- Find username field ---
    const usernameKeywords = ['user', 'username', 'name', 'email', 'login', 'account', 'id', 'phone', 'mobile', 'mail'];
    const isUsernameField = (el) => {
      if (el.tag !== 'input') return false;
      if (el.type === 'password' || el.type === 'submit' || el.type === 'button') return false;
      // autocomplete hint
      const autocomplete = el.autocomplete || '';
      if (autocomplete === 'username' || autocomplete === 'email') return true;
      const text = [
        (el.name || '').toLowerCase(),
        (el.id || '').toLowerCase(),
        (el.placeholder || '').toLowerCase(),
        (el.ariaLabel || '').toLowerCase(),
      ].join(' ');
      return usernameKeywords.some(k => text.includes(k));
    };

    let usernameField = elements.find(isUsernameField);
    if (!usernameField) {
      // Fallback: first text/email input
      usernameField = elements.find(e => e.tag === 'input' && e.type !== 'password' && e.type !== 'hidden' && e.type !== 'submit');
    }
    console.log('[DEBUG] Username field:', !!usernameField, usernameField);

    // --- Find password field ---
    const isPasswordField = (el) => el.tag === 'input' && el.type === 'password';
    let passwordField = elements.find(isPasswordField);
    console.log('[DEBUG] Password field:', !!passwordField, passwordField);

    // --- Find login button ---
    const loginKeywords = ['log', 'sign', 'login', 'signin', 'sign in', 'submit', 'enter', 'go', 'continue', 'next', '登', 'ログ'];
    const isLoginButton = (el) => {
      const text = [
        (el.text || '').toLowerCase(),
        (el.value || '').toLowerCase(),
        (el.ariaLabel || '').toLowerCase(),
        (el.name || '').toLowerCase(),
        (el.id || '').toLowerCase(),
      ].join(' ');
      return loginKeywords.some(k => text.includes(k));
    };
    let loginButton = elements.find(e => (e.tag === 'button' || (e.tag === 'input' && (e.type === 'submit' || e.type === 'button'))) && isLoginButton(e));
    if (!loginButton) {
      // Fallback: any submit button
      loginButton = elements.find(e => (e.tag === 'button' && e.type === 'submit') || (e.tag === 'input' && e.type === 'submit'));
    }
    console.log('[DEBUG] Login button:', !!loginButton, loginButton);

    // --- Build actions ---
    if (usernameField) {
      actions.push({ type: 'click', x: usernameField.position.x, y: usernameField.position.y, description: 'Focus username' });
      actions.push({ type: 'type', text: username, selector: usernameField.selector, description: `Type "${username}"` });
      actions.push({ type: 'wait', ms: 300, description: 'Wait' });
    }

    if (passwordField) {
      actions.push({ type: 'click', x: passwordField.position.x, y: passwordField.position.y, description: 'Focus password' });
      actions.push({ type: 'type', text: password, selector: passwordField.selector, description: 'Type password' });
      actions.push({ type: 'wait', ms: 300, description: 'Wait' });
    }

    if (loginButton) {
      actions.push({ type: 'click', x: loginButton.position.x, y: loginButton.position.y, description: 'Click login' });
    } else {
      actions.push({ type: 'press', key: 'Enter', description: 'Press Enter to submit' });
    }
    return actions;
  }

  // ========================================================================
  // 3. FORM FILL WORKFLOW: "Fill the contact form with name 'X' and email 'Y'"
  // ========================================================================
  if ((promptLower.includes('fill') && (promptLower.includes('form') || promptLower.includes('contact'))) ||
      promptLower.includes('fill out')) {
    // Extract all field=value pairs from the prompt
    // Matches: name 'John Doe', email 'john@example.com', message 'Hello', etc.
    const fieldRegex = /(\w+)\s+["']([^"']+)["']/g;
    const fields = {};
    let m;
    while ((m = fieldRegex.exec(prompt)) !== null) {
      fields[m[1].toLowerCase()] = m[2];
    }
    console.log('[DEBUG] Form fields to fill:', fields);

    // Field → keyword mapping for finding inputs
    const fieldKeywords = {
      name: ['name', 'fullname', 'full_name', 'yourname'],
      email: ['email', 'e-mail', 'mail'],
      message: ['message', 'msg', 'comment', 'body', 'text'],
      phone: ['phone', 'tel', 'telephone', 'mobile', 'cell'],
      subject: ['subject', 'topic', 'title'],
      company: ['company', 'org', 'organization'],
      address: ['address', 'addr'],
    };

    let firstInput = null;

    for (const [fieldName, fieldValue] of Object.entries(fields)) {
      const keywords = fieldKeywords[fieldName] || [fieldName];
      const isTargetField = (el) => {
        if (el.tag !== 'input' && el.tag !== 'textarea') return false;
        if (el.type === 'password' || el.type === 'submit' || el.type === 'button' || el.type === 'hidden') return false;
        const text = [
          (el.name || '').toLowerCase(),
          (el.id || '').toLowerCase(),
          (el.placeholder || '').toLowerCase(),
          (el.ariaLabel || '').toLowerCase(),
          (el.autocomplete || '').toLowerCase(),
        ].join(' ');
        return keywords.some(k => text.includes(k));
      };

      // Special case: email can also match type="email"
      let targetField;
      if (fieldName === 'email') {
        targetField = elements.find(e => e.tag === 'input' && e.type === 'email') ||
                      elements.find(isTargetField);
      } else {
        targetField = elements.find(isTargetField);
      }

      if (targetField) {
        if (!firstInput) firstInput = targetField;
        actions.push({ type: 'click', x: targetField.position.x, y: targetField.position.y, description: `Focus "${fieldName}"` });
        actions.push({ type: 'type', text: fieldValue, selector: targetField.selector, clear: true, description: `Type "${fieldValue}"` });
        actions.push({ type: 'wait', ms: 200, description: 'Wait' });
        console.log(`[DEBUG] Filled field "${fieldName}" with "${fieldValue}"`);
      } else {
        console.log(`[DEBUG] Field "${fieldName}" not found — typing into first input`);
        // Fallback: type into first available input
        if (!firstInput) {
          firstInput = elements.find(e =>
            (e.tag === 'input' && e.type !== 'password' && e.type !== 'hidden' && e.type !== 'submit') ||
            e.tag === 'textarea'
          );
        }
        if (firstInput) {
          actions.push({ type: 'click', x: firstInput.position.x, y: firstInput.position.y, description: `Focus input for "${fieldName}"` });
          actions.push({ type: 'type', text: fieldValue, selector: firstInput.selector, clear: false, description: `Append "${fieldValue}"` });
          actions.push({ type: 'wait', ms: 200, description: 'Wait' });
        }
      }
    }

    // --- Find submit button ---
    const submitKeywords = ['submit', 'send', 'post', 'save', 'confirm', 'apply', 'register', 'sign up', '提交', '送信'];
    const isSubmitButton = (el) => {
      const text = [
        (el.text || '').toLowerCase(),
        (el.value || '').toLowerCase(),
        (el.ariaLabel || '').toLowerCase(),
        (el.name || '').toLowerCase(),
        (el.id || '').toLowerCase(),
      ].join(' ');
      return submitKeywords.some(k => text.includes(k));
    };
    const submitButton = elements.find(e =>
      (e.tag === 'button' || (e.tag === 'input' && (e.type === 'submit' || e.type === 'button'))) &&
      isSubmitButton(e)
    );
    if (!submitButton) {
      const fallback = elements.find(e =>
        (e.tag === 'button' && e.type === 'submit') ||
        (e.tag === 'input' && e.type === 'submit')
      );
      if (fallback) {
        actions.push({ type: 'click', x: fallback.position.x, y: fallback.position.y, description: 'Click submit' });
      } else {
        actions.push({ type: 'press', key: 'Enter', description: 'Press Enter to submit' });
      }
    } else {
      actions.push({ type: 'click', x: submitButton.position.x, y: submitButton.position.y, description: 'Click submit' });
    }
    return actions;
  }

  // ========================================================================
  // 4. SCROLL WORKFLOW: "Scroll down/up the page and capture the page state"
  // ========================================================================
  if (promptLower.includes('scroll')) {
    const direction = promptLower.includes('up') ? 'up' : 'down';
    const amountMatch = prompt.match(/scroll\s+(?:down|up)\s+(?:by\s+)?(\d+)/i);
    const amount = amountMatch ? parseInt(amountMatch[1]) : 500;
    console.log('[DEBUG] Scroll action:', { direction, amount });

    actions.push({ type: 'scroll', direction, amount, description: `Scroll ${direction} ${amount}px` });
    actions.push({ type: 'wait', ms: 1000, description: 'Wait for scroll to settle' });
    actions.push({ type: 'getState', description: 'Capture page state after scroll' });
    return actions;
  }

  // ========================================================================
  // 5. SHOW / LIST elements — no action needed; executeWorkflow Step 1
  //    already logged all interactive elements via getState().
  //    Must be checked BEFORE "click" branch because prompts like
  //    "Show me all clickable elements" contain the word "click".
  // ========================================================================
  if (promptLower.includes('show me') || promptLower.includes('list all') || promptLower.includes('display all')) {
    actions._displayOnly = true;
    return actions; // empty — getState() in Step 1 already displayed elements
  }

  // ========================================================================
  // 6. CLICK WORKFLOW: "Click the first link on the page"
  // ========================================================================
  if (promptLower.includes('click') || promptLower.includes('press')) {
    const firstLink = elements.find(e => e.tag === 'a' && e.text && e.text.trim().length > 0);
    if (firstLink) {
      actions.push({ type: 'click', x: firstLink.position.x, y: firstLink.position.y, description: `Click: ${firstLink.text.slice(0, 40)}` });
      return actions;
    }
  }

  // ========================================================================
  // 7. FALLBACK: click the first element
  // ========================================================================
  if (elements.length > 0) {
    const firstEl = elements[0];
    actions.push({ type: 'click', x: firstEl.position.x, y: firstEl.position.y, description: `Click: ${firstEl.text?.slice(0, 40) || firstEl.tag}` });
  }

  return actions;
}

function findMatchingElement(target, elements) {
  const targetLower = target.toLowerCase();

  // Exact text match (highest priority)
  for (const el of elements) {
    if (el.text?.toLowerCase() === targetLower) return el;
  }

  // Contains match
  for (const el of elements) {
    if (el.text?.toLowerCase().includes(targetLower)) return el;
  }

  // ID/name match
  for (const el of elements) {
    if (el.id?.toLowerCase().includes(targetLower) || el.name?.toLowerCase().includes(targetLower)) {
      return el;
    }
  }

  // href match for links
  for (const el of elements) {
    if (el.href?.toLowerCase().includes(targetLower)) return el;
  }

  return null;
}

// ---------------------------------------------------------------------------
// Workflow Loading
// ---------------------------------------------------------------------------

async function loadWorkflows() {
  try {
    const response = await fetch('../workflows/sample_workflows.json');
    const data = await response.json();
    state.workflows = data.workflows || [];

    state.workflows.forEach((w) => {
      const option = document.createElement('option');
      option.value = w.id;
      option.textContent = w.name;
      state.$workflowSelect.appendChild(option);
    });
  } catch (err) {
    logMessage('warn', `Could not load workflows: ${err.message}`);
    // Add some defaults
    state.workflows = [
      { id: 'default-1', name: 'Click first link', prompt: 'Click the first link on the page' },
      { id: 'default-2', name: 'Search and navigate', prompt: 'Search for "AI news" and click the first result' },
    ];
    state.workflows.forEach((w) => {
      const option = document.createElement('option');
      option.value = w.id;
      option.textContent = w.name;
      state.$workflowSelect.appendChild(option);
    });
  }
}

// ---------------------------------------------------------------------------
// UI Helpers
// ---------------------------------------------------------------------------

function updateStatus(status, message) {
  state.$statusBar.className = `status ${status}`;
  state.$statusBar.textContent = message;
}

function logMessage(level, message) {
  const timestamp = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = `log-entry log-${level}`;
  entry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${escapeHtml(message)}`;
  state.$logOutput.appendChild(entry);
  state.$logOutput.scrollTop = state.$logOutput.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Dump interactive elements to the UI log as a compact table.
 * Groups by tag for readability.
 */
function dumpElementsToLog(elements, maxShow = 60) {
  const shown = elements.slice(0, maxShow);
  const tags = {};
  shown.forEach(el => {
    const t = el.tag || '?';
    if (!tags[t]) tags[t] = [];
    tags[t].push(el);
  });

  logMessage('info', `--- Element List (showing ${shown.length} of ${elements.length}) ---`);

  for (const [tag, els] of Object.entries(tags)) {
    const rows = els.map(el => {
      const label = el.text || el.placeholder || el.ariaLabel || el.value || el.name || el.id || '';
      const detail = el.type ? `[${el.type}]` : '';
      const pos = el.position ? `(${el.position.x},${el.position.y})` : '';
      return `${detail} "${label.slice(0, 50)}" ${pos}`.trim();
    });
    logMessage('info', `  ${tag} (${els.length}): ${rows.join(' | ')}`);
  }

  if (elements.length > maxShow) {
    logMessage('info', `  ... and ${elements.length - maxShow} more (see DevTools console for full list)`);
  }
}
