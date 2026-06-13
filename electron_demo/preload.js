/**
 * Secure preload script for the control UI.
 * Exposes only the necessary APIs via contextBridge.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Sandbox control
  navigate: (url) => ipcRenderer.invoke('sandbox:navigate', url),
  execute: (action) => ipcRenderer.invoke('sandbox:execute', action),
  screenshot: () => ipcRenderer.invoke('sandbox:screenshot'),
  getSandboxInfo: () => ipcRenderer.invoke('sandbox:info'),
  reloadSandbox: () => ipcRenderer.invoke('sandbox:reload'),

  // Event listeners
  onSandboxLog: (callback) => {
    ipcRenderer.on('sandbox:log', (_event, data) => callback(data));
  },
  onSandboxNavigated: (callback) => {
    ipcRenderer.on('sandbox:navigated', (_event, url) => callback(url));
  },

  // Agent bridge
  sendToAgent: (message) => ipcRenderer.invoke('agent:send', message),
  onAgentMessage: (callback) => {
    ipcRenderer.on('agent:message', (_event, data) => callback(data));
  },

  // LLM proxy
  llmCall: (params) => ipcRenderer.invoke('llm:fetch', params),
});
