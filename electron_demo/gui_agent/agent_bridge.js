/**
 * Agent Bridge — Communication layer between Electron control panel and Python backend.
 *
 * Uses WebSocket to send GUI agent commands and receive responses.
 * Default backend: ws://localhost:8765
 */

const { ipcMain } = require('electron');
const WebSocket = require('ws');

const DEFAULT_WS_URL = 'ws://localhost:8765';

/**
 * @typedef {Object} AgentMessage
 * @property {string} type — 'predict' | 'execute' | 'status'
 * @property {Object} payload — depends on type
 */

/**
 * @typedef {Object} AgentResponse
 * @property {boolean} success
 * @property {Object} [data]
 * @property {string} [error]
 */

class AgentBridge {
  constructor(wsUrl = DEFAULT_WS_URL) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.connected = false;
    this.pendingRequests = new Map();
    this.requestId = 0;
  }

  /**
   * Connect to the Python GUI Agent backend.
   */
  connect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.on('open', () => {
          this.connected = true;
          console.log('[AgentBridge] Connected to Python backend');
          resolve(true);
        });

        this.ws.on('message', (data) => {
          try {
            const response = JSON.parse(data.toString());
            const { requestId, ...rest } = response;

            if (requestId && this.pendingRequests.has(requestId)) {
              const { resolve: res } = this.pendingRequests.get(requestId);
              this.pendingRequests.delete(requestId);
              res(rest);
            }
          } catch (e) {
            console.error('[AgentBridge] Failed to parse message:', e);
          }
        });

        this.ws.on('close', () => {
          this.connected = false;
          console.log('[AgentBridge] Disconnected');
        });

        this.ws.on('error', (err) => {
          console.error('[AgentBridge] WebSocket error:', err.message);
          this.connected = false;
          reject(err);
        });
      } catch (e) {
        reject(e);
      }
    });
  }

  /**
   * Send a message and wait for the response.
   * @param {AgentMessage} message
   * @returns {Promise<AgentResponse>}
   */
  send(message) {
    return new Promise((resolve, reject) => {
      if (!this.connected || !this.ws) {
        reject(new Error('Not connected to agent backend'));
        return;
      }

      const id = ++this.requestId;
      const payload = { requestId: id, ...message };

      this.pendingRequests.set(id, { resolve, reject });

      try {
        this.ws.send(JSON.stringify(payload));
      } catch (e) {
        this.pendingRequests.delete(id);
        reject(e);
      }

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error('Request timed out'));
        }
      }, 30000);
    });
  }

  /**
   * Send a predict request to the GUI agent.
   * @param {string} imageBase64 — screenshot as base64
   * @param {string} instruction — user prompt
   * @returns {Promise<Object>} parsed action
   */
  async predict(imageBase64, instruction) {
    return this.send({
      type: 'predict',
      payload: { image: imageBase64, instruction },
    });
  }

  /**
   * Check backend status.
   */
  async status() {
    return this.send({ type: 'status', payload: {} });
  }

  /**
   * Close the connection.
   */
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.connected = false;
    }
  }
}

// Singleton
let bridgeInstance = null;

function getBridge() {
  if (!bridgeInstance) {
    bridgeInstance = new AgentBridge();
  }
  return bridgeInstance;
}

module.exports = { AgentBridge, getBridge };
