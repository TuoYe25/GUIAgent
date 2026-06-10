/**
 * Edge Device GUI Agent — Electron Main Process
 *
 * Manages BrowserWindow for the control UI and BrowserView for the sandboxed
 * target website where GUI agent actions are executed.
 */

const { app, BrowserWindow, BrowserView, ipcMain, session, nativeImage } = require('electron');
const path = require('path');
const fs = require('fs');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** @type {BrowserWindow} */
let mainWindow = null;

/** @type {BrowserView} */
let sandboxView = null;

const DEFAULT_TARGET_URL = 'https://example.com';

// ---------------------------------------------------------------------------
// App Lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(() => {
  createMainWindow();
  createSandboxView();
  registerIPC();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// ---------------------------------------------------------------------------
// Main Window (Control UI)
// ---------------------------------------------------------------------------

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    title: 'Edge GUI Agent Demo',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ---------------------------------------------------------------------------
// Sandbox BrowserView
// ---------------------------------------------------------------------------

function createSandboxView() {
  sandboxView = new BrowserView({
    webPreferences: {
      contextIsolation: true, // 必须为 true 才能使用 preload
      nodeIntegration: false,
      sandbox: false, // 需要 preload 进行动作注入
      preload: path.join(__dirname, 'gui_agent', 'action_executor.js'),
      webSecurity: true,
      allowRunningInsecureContent: false,
    },
  });

  mainWindow?.addBrowserView(sandboxView);
  resizeSandboxView();

  // Capture sandbox console logs
  sandboxView.webContents.on('console-message', (event, level, message) => {
    if (mainWindow) {
      mainWindow.webContents.send('sandbox:log', { level: level, message });
    }
  });

  // Capture sandbox navigations
  sandboxView.webContents.on('did-navigate', (event, url) => {
    if (mainWindow) {
      mainWindow.webContents.send('sandbox:navigated', url);
    }
  });

  // Load default page
  sandboxView.webContents.loadURL(DEFAULT_TARGET_URL);
}

function resizeSandboxView() {
  if (!mainWindow || !sandboxView) return;

  const [width, height] = mainWindow.getSize();

  // Control panel on the left (400px), sandbox on the right
  const controlWidth = 380;
  const sandboxWidth = width - controlWidth;

  sandboxView.setBounds({
    x: controlWidth,
    y: 0,
    width: sandboxWidth,
    height: height,
  });
}

// ---------------------------------------------------------------------------
// IPC Handlers
// ---------------------------------------------------------------------------

function registerIPC() {
  // Resize
  if (mainWindow) {
    mainWindow.on('resize', () => {
      resizeSandboxView();
    });
  }

  // Navigate sandbox to URL
  ipcMain.handle('sandbox:navigate', async (event, url) => {
    try {
      if (!sandboxView) return { success: false, error: 'No sandbox view' };
      await sandboxView.webContents.loadURL(url);
      return { success: true, url };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // Execute action in sandbox
  ipcMain.handle('sandbox:execute', async (event, action) => {
    try {
      if (!sandboxView) return { success: false, error: 'No sandbox view' };

      const result = await sandboxView.webContents.executeJavaScript(
        `window.__guiAgent_execute(${JSON.stringify(action)})`
      );
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // Capture screenshot of sandbox
  ipcMain.handle('sandbox:screenshot', async () => {
    try {
      if (!sandboxView) return { success: false, error: 'No sandbox view' };

      const image = await sandboxView.webContents.capturePage();
      const base64 = image.toDataURL();
      
      // 保存截图到文件
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const screenshotPath = path.join(app.getPath('desktop'), `gui-agent-screenshot-${timestamp}.png`);
      
      const buffer = nativeImage.createFromDataURL(base64).toPNG();
      fs.writeFileSync(screenshotPath, buffer);
      
      return { success: true, image: base64, filePath: screenshotPath };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // Get sandbox page info
  ipcMain.handle('sandbox:info', async () => {
    try {
      if (!sandboxView) return { success: false, error: 'No sandbox view' };

      const info = await sandboxView.webContents.executeJavaScript(`
        (() => ({
          url: window.location.href,
          title: document.title,
          readyState: document.readyState,
          viewport: { width: window.innerWidth, height: window.innerHeight }
        }))()
      `);
      return { success: true, info };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // Reload sandbox
  ipcMain.handle('sandbox:reload', async () => {
    try {
      if (!sandboxView) return { success: false, error: 'No sandbox view' };
      sandboxView.webContents.reload();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // LLM API proxy — renderer can't call external APIs due to CSP
  ipcMain.handle('llm:fetch', async (event, { endpoint, apiKey, model, body }) => {
    const url = new URL(endpoint);
    const isHttps = url.protocol === 'https:';
    const http = isHttps ? require('https') : require('http');

    return new Promise((resolve) => {
      const postData = JSON.stringify(body);
      const options = {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname + url.search,
        method: 'POST',
        rejectUnauthorized: false,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'Content-Length': Buffer.byteLength(postData),
        },
        timeout: 30000,
      };

      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            resolve({ success: true, data });
          } else {
            resolve({ success: false, error: `HTTP ${res.statusCode}: ${data.slice(0, 300)}` });
          }
        });
      });

      req.on('error', (err) => {
        resolve({ success: false, error: err.message });
      });

      req.on('timeout', () => {
        req.destroy();
        resolve({ success: false, error: 'Request timeout (30s)' });
      });

      req.write(postData);
      req.end();
    });
  });
}
