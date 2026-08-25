# 旅游出行小帮手

一个本地中文旅游出行智能体：使用 LangGraph 执行“计划—行动—观察—校验”循环，DeepSeek 理解目标和生成结构化计划，高德 Web 服务提供真实天气、地点和路线数据，并用 SQLite 保存可持续修改的行程与用户主动保存的偏好。

## 主要功能

- 在同一个对话中查询天气、创建多日行程并继续追问。
- 使用高德查询行政区、天气、景点、餐饮和活动间路线。
- 支持“第二天呢”“只改第二天”“换成亲子景点”等上下文追问和局部修改。
- 天气可能影响行程时先展示调整建议，由用户确认后再保存。
- 保存最近 20 轮行程消息、行程摘要、历史版本和用户明确要求记住的偏好。
- 支持查看、恢复和永久删除行程。
- 在安全执行预算内自主选择下一步工具，并根据地点歧义、工具结果和行程校验决定继续、补问或停止。
- 每个天气、地点和路线结果都携带高德来源、资源标识和查询时间。

## 技术架构

| 层级 | 技术 | 用途 |
| --- | --- | --- |
| 前端 | React、TypeScript、Vinext | 聊天、行程卡片、状态流和高德交互地图 |
| 后端 | FastAPI、Pydantic | 统一消息接口、数据校验和 SSE 流式输出 |
| 任务编排 | LangGraph、DeepSeek | 目标识别、结构化规划、受控工具循环和上下文理解 |
| 数据工具 | 高德 Web 服务 API | 地理编码、天气、POI 搜索和路径规划 |
| 持久化 | SQLite | 行程、版本、消息、摘要、调整建议和偏好 |

## 项目目录

```text
Agent/
├─ backend/
│  ├─ app/
│  │  ├─ main.py              # FastAPI 应用与公开接口
│  │  ├─ unified_agent.py     # 天气与旅行统一入口
│  │  ├─ travel_agent.py      # 多日行程编排、路线和天气调整
│  │  ├─ itinerary_engine.py  # 时间段、区域聚类和约束校验
│  │  ├─ date_parser.py       # 中文日期、天数和目标日期解析
│  │  ├─ tools/               # 带类型、错误码和来源的正式工具层
│  │  ├─ agent.py             # 独立天气查询能力
│  │  ├─ amap.py              # 高德 API、重试和缓存
│  │  ├─ db.py                # SQLite 数据访问与版本管理
│  │  ├─ models.py            # Pydantic 数据模型
│  │  └─ config.py            # 后端环境配置
│  ├─ tests/
│  │  ├─ test_agent.py
│  │  ├─ test_amap.py
│  │  ├─ test_api.py
│  │  ├─ test_travel_agent.py
│  │  ├─ test_unified_agent.py
│  │  └─ test_date_parser.py
│  ├─ .env.example            # 后端配置示例，不含真实密钥
│  └─ pyproject.toml
├─ frontend/
│  ├─ app/
│  │  ├─ page.tsx             # 统一聊天与三栏行程界面
│  │  ├─ globals.css          # 页面样式和响应式布局
│  │  └─ layout.tsx
│  ├─ lib/
│  │  ├─ sse.ts               # SSE 事件与前端数据类型
│  │  └─ sse.test.ts
│  ├─ public/                 # 静态资源
│  ├─ .env.example            # 地图配置示例，不含真实密钥
│  └─ package.json
├─ setup.ps1                  # 安装前后端依赖
├─ start.ps1                  # 启动前后端并记录项目 PID
├─ restart.ps1                # 安全重启本项目服务
├─ .gitignore                 # 排除密钥、数据库和构建产物
└─ README.md
```

## 快速开始

环境要求：Python 3.11+、Node.js 22.13+，以及 DeepSeek、高德开放平台的 API Key。

```powershell
.\setup.ps1
```

编辑 `backend/.env`：

```dotenv
DEEPSEEK_API_KEY=你的密钥
AMAP_WEB_API_KEY=你的高德Web服务API密钥
```

如需交互地图，在 `frontend/.env` 中配置：

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_AMAP_JS_KEY=你的高德JSAPIKey
NEXT_PUBLIC_AMAP_SECURITY_CODE=你的高德JSAPI安全密钥
```

启动项目：

```powershell
.\start.ps1
```

修改密钥或代码后可安全重启：

```powershell
.\restart.ps1
```

打开 <http://localhost:3000>。后端 API 文档位于 <http://localhost:8000/docs>。

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/assistant/messages` | 统一对话入口，以 SSE 返回步骤和回答 |
| `GET` | `/api/conversations/{id}/messages` | 获取统一对话最近消息 |
| `GET` | `/api/trips` | 查看已保存行程 |
| `GET` | `/api/trips/{id}` | 获取最新行程 |
| `DELETE` | `/api/trips/{id}` | 删除行程及关联数据 |
| `GET` | `/api/trips/{id}/messages` | 获取最近行程消息 |
| `GET` | `/api/trips/{id}/versions` | 查看历史版本 |
| `POST` | `/api/trips/{id}/proposals/{proposal_id}/apply` | 确认调整建议 |
| `GET` | `/api/profile` | 获取天气偏好 |
| `GET` | `/api/travel-profile` | 获取旅行偏好 |
| `GET` | `/api/health` | 查看服务和密钥配置状态 |

## 数据边界

- 天气、地点和路线事实只来自高德接口。
- 不提供逐小时、长期、历史天气、空气质量和灾害预警。
- 酒店、门票、车票和餐饮价格不包含实时库存或预订能力。
- 跨行程偏好只保存用户明确要求记住的内容。
- `.env`、SQLite 数据库、运行 PID 和构建产物不会提交到 Git。

## 验证

```powershell
& .\.venv\Scripts\python.exe -m pytest backend

cd frontend
npm run test
npm run lint
npm run build
```

自动化测试默认使用模拟响应，不需要真实密钥。
