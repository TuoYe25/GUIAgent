/**
 * Sandbox Action Executor — injected into the BrowserView.
 *
 * Provides window.__guiAgent_execute(action) via contextBridge for:
 *   - click(x, y)
 *   - type(selector, text)
 *   - scroll(direction)
 *   - navigate(url)
 *   - wait(ms)
 *   - select(selector, value)
 *   - hover(x, y)
 *   - press(key)
 *   - getState() — returns DOM info for the agent
 */

const { contextBridge } = require('electron');

(function () {
  'use strict';

  const actionHandlers = {
    /**
     * Click at coordinates or on a selector.
     * @param {{x?: number, y?: number, selector?: string}} action
     */
    click(action) {
      const target = resolveTarget(action);
      if (!target) return { error: 'No click target resolved' };

      const eventOpts = { bubbles: true, cancelable: true, view: window };
      target.dispatchEvent(new MouseEvent('mousedown', eventOpts));
      target.dispatchEvent(new MouseEvent('mouseup', eventOpts));
      target.dispatchEvent(new MouseEvent('click', eventOpts));

      // Also focus the element
      if (target.focus) target.focus();

      return { success: true, action: 'click', tag: target.tagName };
    },

    /**
     * Type text into an element. Uses native input setter for React/Vue compat.
     * @param {{selector?: string, text: string, clear?: boolean}} action
     */
    type(action) {
      const el = action.selector
        ? document.querySelector(action.selector)
        : document.activeElement;

      if (!el) return { error: 'No element to type into' };

      el.focus();

      // Use native setter so React/Vue controlled inputs pick up the change
      const nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
      )?.set || Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      )?.set;

      const newValue = action.clear ? action.text : (el.value || '') + action.text;

      if (nativeSetter) {
        nativeSetter.call(el, newValue);
      } else {
        el.value = newValue;
      }

      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));

      return { success: true, action: 'type', text: action.text };
    },

    /**
     * Scroll the page or a specific element.
     * @param {{direction: string, selector?: string, amount?: number}} action
     */
    scroll(action) {
      const el = action.selector
        ? document.querySelector(action.selector)
        : window;

      const amount = action.amount || 300;
      const direction = action.direction === 'up' ? -amount : amount;

      if (el === window) {
        window.scrollBy({ top: direction, behavior: 'smooth' });
      } else if (el) {
        el.scrollBy({ top: direction, behavior: 'smooth' });
      }

      return { success: true, action: 'scroll', direction: action.direction };
    },

    /**
     * Navigate the sandbox to a different URL.
     * @param {{url: string}} action
     */
    navigate(action) {
      window.location.href = action.url;
      return { success: true, action: 'navigate', url: action.url };
    },

    /**
     * Wait for a specified duration or for a selector to appear.
     * @param {{ms?: number, selector?: string, timeout?: number}} action
     */
    async wait(action) {
      if (action.ms) {
        await new Promise((r) => setTimeout(r, action.ms));
        return { success: true, action: 'wait', ms: action.ms };
      }

      if (action.selector) {
        const timeout = action.timeout || 5000;
        const result = await waitForSelector(action.selector, timeout);
        return result;
      }

      return { error: 'wait requires ms or selector' };
    },

    /**
     * Select an option from a <select> element.
     * @param {{selector: string, value: string}} action
     */
    select(action) {
      const el = document.querySelector(action.selector);
      if (!el) return { error: `Selector not found: ${action.selector}` };
      if (el.tagName !== 'SELECT') return { error: 'Element is not a <select>' };

      el.value = action.value;
      el.dispatchEvent(new Event('change', { bubbles: true }));

      return { success: true, action: 'select', value: action.value };
    },

    /**
     * Hover at coordinates or on a selector.
     * @param {{x?: number, y?: number, selector?: string}} action
     */
    hover(action) {
      const target = resolveTarget(action);
      if (!target) return { error: 'No hover target resolved' };

      const eventOpts = { bubbles: true, cancelable: true, view: window };
      target.dispatchEvent(new MouseEvent('mouseenter', eventOpts));
      target.dispatchEvent(new MouseEvent('mouseover', eventOpts));

      return { success: true, action: 'hover', tag: target.tagName };
    },

    /**
     * Press a keyboard key on the active element.
     * Includes legacy keyCode / which for wider website compatibility.
     * @param {{key: string}} action
     */
    press(action) {
      const el = document.activeElement;
      const keyCodeMap = {
        Enter: 13, Tab: 9, Escape: 27, Backspace: 8,
        ArrowUp: 38, ArrowDown: 40, ArrowLeft: 37, ArrowRight: 39,
        Space: 32, Delete: 46,
      };
      const keyCode = keyCodeMap[action.key] || 0;

      const eventOpts = {
        bubbles: true,
        cancelable: true,
        key: action.key,
        code: action.key,
        keyCode,
        which: keyCode,
      };

      if (el) {
        el.dispatchEvent(new KeyboardEvent('keydown', eventOpts));
        el.dispatchEvent(new KeyboardEvent('keypress', eventOpts));
        el.dispatchEvent(new KeyboardEvent('keyup', eventOpts));
      }

      // If Enter was pressed on an input inside a form, try submitting the form
      if (action.key === 'Enter' && el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
        const form = el.closest('form');
        if (form) {
          try {
            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
          } catch (_) { /* ignore */ }
        }
      }

      return { success: true, action: 'press', key: action.key };
    },

    /**
     * Get the current state of the page (DOM info for the agent).
     */
    getState() {
      const interactiveElements = [];

      document.querySelectorAll('a, button, input, select, textarea, [role="button"], [onclick]')
        .forEach((el) => {
          const rect = el.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) return; // invisible

          interactiveElements.push({
            tag: el.tagName.toLowerCase(),
            type: el.type || null,
            id: el.id || null,
            name: el.getAttribute('name') || null,
            placeholder: el.getAttribute('placeholder') || null,
            value: el.value || el.getAttribute('value') || null,
            text: (el.textContent || '').trim().slice(0, 100),
            ariaLabel: el.getAttribute('aria-label') || null,
            autocomplete: el.getAttribute('autocomplete') || null,
            href: el.tagName === 'A' ? el.href : null,
            visible: true,
            position: {
              x: Math.round(rect.x + rect.width / 2),
              y: Math.round(rect.y + rect.height / 2),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            },
            selector: buildSelector(el),
          });
        });

      return {
        success: true,
        action: 'getState',
        page: {
          url: window.location.href,
          title: document.title,
          viewport: { width: window.innerWidth, height: window.innerHeight },
          scrollY: window.scrollY,
          bodyText: (document.body?.textContent || '').trim().slice(0, 3000),
        },
        interactiveElements,
      };
    },
  };

  // ---------------------------------------------------------------------------
  // Resolve click/hover target from either (x,y) or selector
  // ---------------------------------------------------------------------------

  function resolveTarget(action) {
    if (action.selector) {
      return document.querySelector(action.selector);
    }
    if (action.x !== undefined && action.y !== undefined) {
      return document.elementFromPoint(action.x, action.y);
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Wait for a selector to appear in the DOM
  // ---------------------------------------------------------------------------

  function waitForSelector(selector, timeout) {
    return new Promise((resolve) => {
      const start = Date.now();

      const el = document.querySelector(selector);
      if (el) {
        resolve({ success: true, action: 'wait', selector, found: true, ms: 0 });
        return;
      }

      const observer = new MutationObserver(() => {
        const el = document.querySelector(selector);
        if (el) {
          observer.disconnect();
          const elapsed = Date.now() - start;
          resolve({ success: true, action: 'wait', selector, found: true, ms: elapsed });
        }
      });

      observer.observe(document.body, { childList: true, subtree: true });

      setTimeout(() => {
        observer.disconnect();
        const found = !!document.querySelector(selector);
        resolve({ success: found, action: 'wait', selector, found, ms: timeout });
      }, timeout);
    });
  }

  // ---------------------------------------------------------------------------
  // Build a CSS selector for an element
  // ---------------------------------------------------------------------------

  function buildSelector(el) {
    if (el.id) return `#${CSS.escape(el.id)}`;

    const parts = [];
    let current = el;

    while (current && current !== document.body && current !== document.documentElement) {
      let segment = current.tagName.toLowerCase();

      if (current.id) {
        parts.unshift(`#${CSS.escape(current.id)}`);
        break;
      }

      if (current.className && typeof current.className === 'string') {
        const classes = current.className.trim().split(/\s+/).slice(0, 2);
        if (classes.length) {
          segment += '.' + classes.map((c) => CSS.escape(c)).join('.');
        }
      }

      // Add nth-child if needed
      const parent = current.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter(
          (s) => s.tagName === current.tagName
        );
        if (siblings.length > 1) {
          const idx = siblings.indexOf(current) + 1;
          segment += `:nth-child(${idx})`;
        }
      }

      parts.unshift(segment);
      current = current.parentElement;
    }

    return parts.join(' > ');
  }

  // ---------------------------------------------------------------------------
  // Main execute function
  // ---------------------------------------------------------------------------

  const __guiAgent_execute = async function (action) {
    console.log('[GUI Agent] __guiAgent_execute called with:', action);
    const { type, ...params } = action;
    const handler = actionHandlers[type];

    if (!handler) {
      return { error: `Unknown action type: ${type}`, available: Object.keys(actionHandlers) };
    }

    try {
      const result = await handler(params);
      console.log('[GUI Agent] Action result:', result);
      return result;
    } catch (error) {
      console.error('[GUI Agent] Action error:', error);
      return { error: error.message, stack: error.stack };
    }
  };

  // Expose to the renderer via contextBridge
  if (typeof contextBridge !== 'undefined') {
    contextBridge.exposeInMainWorld('__guiAgent_execute', __guiAgent_execute);
    console.log('[GUI Agent] Action executor exposed via contextBridge');
  } else {
    // Fallback for non-context-isolation mode
    window.__guiAgent_execute = __guiAgent_execute;
    console.log('[GUI Agent] Action executor exposed to window (no context isolation)');
  }
})();
