# GUI Agent — Electron Sandbox Demo

基于 Electron 的 GUI 自动化 Agent，通过自然语言指令操控浏览器页面。

## 架构

```
electron_demo/
├── main.js              # Electron 主进程（BrowserWindow + BrowserView 沙箱）
├── preload.js           # contextBridge 安全桥接
├── renderer/
│   ├── index.html       # 控制面板 UI
│   ├── app.js           # Agent 核心逻辑（解析器 + LLM/VLM 调度）
│   └── styles.css       # 样式
├── gui_agent/
│   ├── action_executor.js   # 浏览器内 action 执行器（注入沙箱）
│   └── agent_bridge.js      # 执行桥接
└── workflows/
    └── sample_workflows.json
```

## 快速启动

```bash
npm install
npm start
```

## 模型 / 解析器

支持三种模式，通过 UI 下拉切换：

| 模式 | 实现 | 说明 |
|------|------|------|
| LLM | DeepSeek v4 (DMXAPI) | 文本模型，传元素 JSON 规划动作 |
| VLM | Qwen-VL-Max (阿里云百炼) | 视觉模型，传截图 + 元素 JSON |
| Regex | 内置 parsePromptToActions | 离线正则解析，无需 API |

VLM 配置位于 `renderer/app.js` 顶部的 `VLM_CONFIG`，使用 DashScope OpenAI 兼容接口。

## 依赖

- Electron ^30
- 无需其他运行时依赖
