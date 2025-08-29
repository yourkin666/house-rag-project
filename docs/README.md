# 房源 RAG 知识库项目

这是一个基于检索增强生成（RAG）技术的房源信息查询系统，使用 Google Gemini 作为大语言模型，PostgreSQL + pgvector 作为向量数据库。

## 🏗️ 项目架构

```
房源 RAG 知识库
├── 数据库层：PostgreSQL + pgvector
├── 向量化层：Google Gemini Embeddings  
├── 检索层：LangChain + PGVector
├── 生成层：Google Gemini Pro
└── API层：FastAPI
    ├── /ask     - 智能房源查询
    ├── /append  - 添加新房源数据
    ├── /health  - 健康检查
    └── /docs    - API文档
```

## 📁 项目结构

```
house-rag-project/
├── app.py                 # FastAPI 应用主程序
├── ingest.py             # 数据向量化脚本
├── requirements.txt      # Python 依赖包
├── Dockerfile           # Docker 镜像构建文件
├── docker-compose.yml   # Docker 服务编排文件
├── init_db.sql          # 数据库初始化脚本
├── sample_data.sql      # 示例数据插入脚本
├── .env.example         # 环境变量模板
├── 计划.md              # 项目开发计划
└── README.md            # 项目说明文档（本文件）
```

## 🚀 快速开始

### 第一步：环境准备

1. **安装 Docker Desktop**
   - 访问 [Docker 官网](https://www.docker.com/products/docker-desktop/) 下载并安装
   - 启动 Docker Desktop 并确保其正常运行

2. **获取 Google AI API Key**
   - 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
   - 登录 Google 账号并创建 API Key
   - 保存生成的 API Key，稍后会用到

3. **克隆或下载项目**
   ```bash
   # 如果你有 git，可以克隆项目
   git clone <your-repo-url>
   cd house-rag-project
   
   # 或者直接下载项目文件到 house-rag-project 目录
   ```

### 第二步：配置环境变量

1. **创建 .env 文件**
   ```bash
   # 在项目根目录创建 .env 文件
   cp .env.example .env
   ```

2. **编辑 .env 文件**
   ```bash
   # 使用你喜欢的编辑器打开 .env 文件
   nano .env
   # 或
   vim .env
   # 或使用图形化编辑器
   ```

3. **填入你的配置信息**
   ```env
   # .env 文件内容示例
   
   # Google AI API Key（必填）
   GOOGLE_API_KEY="你的真实API密钥"
   
   # 数据库配置（可自定义）
   DB_HOST="localhost"
   DB_PORT="5433"
   DB_NAME="house_knowledge_base"
   DB_USER="houseuser"
   DB_PASSWORD="securepassword123"
   ```

### 第三步：启动服务

1. **构建并启动所有服务**
   ```bash
   # 在项目根目录执行
   docker-compose up -d --build
   ```

2. **检查服务状态**
   ```bash
   # 查看运行中的容器
   docker ps
   
   # 应该看到两个容器正在运行：
   # - house_rag_db (PostgreSQL 数据库)
   # - house_rag_app (Python 应用容器)
   ```

3. **查看服务日志**（可选）
   ```bash
   # 查看数据库日志
   docker-compose logs db
   
   # 查看应用日志
   docker-compose logs app
   ```

### 第四步：初始化数据库

1. **连接数据库并初始化表结构**
   
   **方法A：使用数据库管理工具（推荐）**
   - 下载 [DBeaver](https://dbeaver.io/) 或 [TablePlus](https://tableplus.com/)
   - 创建新的 PostgreSQL 连接：
     - 主机：`localhost`
     - 端口：`5433`（或你在 .env 中设置的端口）
     - 数据库：`house_knowledge_base`
     - 用户名和密码：按 .env 文件中的配置
   - 连接成功后，执行 `init_db.sql` 文件中的所有 SQL 命令

   **方法B：使用命令行**
   ```bash
   # 将初始化脚本复制到数据库容器中并执行
   docker cp init_db.sql house_rag_db:/tmp/init_db.sql
   docker-compose exec db psql -U houseuser -d house_knowledge_base -f /tmp/init_db.sql
   ```

2. **插入示例数据**
   ```bash
   # 使用数据库管理工具执行 sample_data.sql
   # 或使用命令行：
   docker cp sample_data.sql house_rag_db:/tmp/sample_data.sql
   docker-compose exec db psql -U houseuser -d house_knowledge_base -f /tmp/sample_data.sql
   ```

### 第五步：生成向量数据

1. **运行数据向量化脚本**
   ```bash
   # 进入应用容器并运行向量化脚本
   docker-compose exec app python ingest.py
   ```

2. **预期输出**
   ```
   🏠 房源数据向量化脚本启动
   ==================================================
   📋 正在加载配置...
   🔌 正在连接数据库...
   ✅ 成功连接到数据库: localhost:5433/house_knowledge_base
   📊 发现 12 条待处理的房源数据
   📝 正在准备向量化文本内容...
   🤖 正在使用 Google Gemini 生成向量...
   📝 处理 12 条文本内容
   ✅ 成功生成 12 个向量
   📐 向量维度: 768
   💾 正在更新数据库...
   ✅ 成功更新 12 条房源的向量数据
   ==================================================
   🎉 向量化处理完成！
   ✨ 成功处理 12 条房源数据
   ```

### 第六步：启动 API 服务

1. **启动 FastAPI 服务**
   ```bash
   # 在应用容器中启动 API 服务
   docker-compose exec app uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **验证服务启动**
   - 访问 [http://localhost:8000](http://localhost:8000) 查看 API 基本信息
   - 访问 [http://localhost:8000/docs](http://localhost:8000/docs) 查看自动生成的 API 文档
   - 访问 [http://localhost:8000/health](http://localhost:8000/health) 进行健康检查

## 🧪 测试和使用

### 1. 使用 Swagger UI 测试（推荐新手）

1. **打开 API 文档页面**
   - 访问 [http://localhost:8000/docs](http://localhost:8000/docs)

2. **测试 /ask 接口（查询房源）**
   - 点击 `/ask` 接口的 "Try it out" 按钮
   - 在请求体中输入：
     ```json
     {
       "question": "我想找一个上海带游泳池的豪华别墅",
       "max_results": 3
     }
     ```
   - 点击 "Execute" 执行请求

3. **测试 /append 接口（添加房源）**
   - 点击 `/append` 接口的 "Try it out" 按钮
   - 在请求体中输入：
     ```json
     {
       "title": "现代简约公寓",
       "location": "上海市徐汇区",
       "price": 680.0,
       "description": "位于徐汇区核心地段的现代简约公寓，面积120平方米，三室两厅两卫，装修精美，家电齐全。小区环境优雅，交通便利，距离地铁站步行5分钟。周边有商场、学校、医院等配套设施。适合家庭居住或投资出租。"
     }
     ```
   - 点击 "Execute" 执行请求

4. **查看响应结果**
   - `/ask` 接口会返回 AI 生成的回答和相关的房源信息
   - `/append` 接口会返回房源添加结果和向量化状态

### 2. 使用 curl 命令测试

```bash
# 测试健康检查
curl -X GET http://localhost:8000/health

# 测试房源查询
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "我想找北京的学区房，价格在500万左右",
       "max_results": 2
     }'

# 添加新房源
curl -X POST "http://localhost:8000/append" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "滨江豪华公寓",
       "location": "杭州市滨江区",
       "price": 520.0,
       "description": "位于钱塘江畔的豪华公寓，面积100平方米，两室两厅一卫，江景房，装修现代时尚。小区配套完善，有游泳池、健身房、儿童游乐区。交通便利，地铁直达。适合年轻白领或小家庭居住。"
     }'

# 获取数据库统计信息
curl -X GET http://localhost:8000/properties/count
```

### 3. 示例查询问题

以下是一些你可以尝试的查询：

```bash
# 地理位置相关
"上海有哪些豪华别墅？"
"北京朝阳区的房源怎么样？"
"三亚的海景房多少钱？"

# 价格相关
"500万以下的房子有哪些？"
"最便宜的房源是什么？"
"2000万左右的豪华房产推荐"

# 功能特点相关
"有游泳池的房子"
"适合养老的房源"
"学区房有哪些选择？"

# 综合查询
"我想要一个上海的高端公寓，预算1000万左右"
"适合年轻人的经济型住房推荐"
```

## 📊 项目监控

### 查看服务状态
```bash
# 查看所有容器状态
docker-compose ps

# 查看资源使用情况
docker stats

# 查看应用日志
docker-compose logs -f app

# 查看数据库日志
docker-compose logs -f db
```

### 数据库监控
```bash
# 进入数据库容器
docker-compose exec db psql -U houseuser -d house_knowledge_base

# 查看数据统计
SELECT * FROM properties_stats;

# 查看所有房源
SELECT id, title, location, price FROM properties LIMIT 10;
```

## 🛠️ 开发和调试

### 进入容器进行调试
```bash
# 进入应用容器
docker-compose exec app bash

# 进入数据库容器
docker-compose exec db bash

# 在容器内可以运行任何 Python 脚本或数据库命令
```

### 重新构建应用
```bash
# 当你修改了代码后，重新构建应用容器
docker-compose up -d --build app

# 或者重启所有服务
docker-compose restart
```

### 查看详细日志
```bash
# 实时查看应用日志
docker-compose logs -f app

# 查看最近的日志
docker-compose logs --tail 100 app
```

## ❗ 常见问题

### 1. 端口冲突
**问题**：`Error: bind: address already in use`

**解决方案**：
- 修改 `.env` 文件中的 `DB_PORT`，改成其他未被占用的端口（如 5434）
- 重新启动服务：`docker-compose down && docker-compose up -d`

### 2. API Key 无效
**问题**：`google.api_core.exceptions.PermissionDenied: 403 API key not valid`

**解决方案**：
- 检查 `.env` 文件中的 `GOOGLE_API_KEY` 是否正确
- 确保 API Key 已启用并有足够的配额
- 重启应用：`docker-compose restart app`

### 3. 数据库连接失败
**问题**：`psycopg2.OperationalError: could not connect to server`

**解决方案**：
- 确保数据库容器正在运行：`docker-compose ps`
- 检查 `.env` 文件中的数据库配置
- 重启数据库服务：`docker-compose restart db`

### 4. 向量化失败
**问题**：运行 `ingest.py` 时出错

**解决方案**：
- 检查网络连接，确保能访问 Google API
- 验证数据库中确实有数据：连接数据库执行 `SELECT COUNT(*) FROM properties;`
- 检查数据库中是否有空的或无效的描述字段

### 5. API 响应慢
**问题**：查询响应时间过长

**解决方案**：
- 检查 Google API 的网络延迟
- 减少 `max_results` 参数值
- 确保数据库索引已正确创建

## 🔧 高级配置

### 修改向量维度
如果你想使用其他的 embedding 模型，需要：

1. 修改 `init_db.sql` 中的向量维度
2. 修改 `ingest.py` 和 `app.py` 中的 embedding 模型配置
3. 删除现有数据并重新初始化

### 添加更多房源数据

**方法一：使用 API 接口（推荐）**
```bash
# 使用 /append 接口，会自动处理向量化
curl -X POST "http://localhost:8000/append" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "你的房源标题",
       "location": "房源地址",
       "price": 500.0,
       "description": "详细的房源描述信息..."
     }'
```

**方法二：直接插入数据库（需手动向量化）**
```sql
-- 在数据库中执行
INSERT INTO properties (title, location, price, description) VALUES 
('你的房源标题', '地址', 价格, '详细描述');

-- 然后运行向量化脚本
docker-compose exec app python ingest.py
```

### 自定义提示词模板
编辑 `app.py` 文件中的 `prompt_template` 部分来自定义 AI 的回答风格。

## 📈 生产部署建议

1. **安全性**：
   - 使用更强的数据库密码
   - 配置防火墙规则
   - 启用 HTTPS

2. **性能优化**：
   - 使用生产级的 PostgreSQL 配置
   - 配置连接池
   - 添加缓存层

3. **监控**：
   - 集成日志收集系统
   - 添加性能监控
   - 设置告警机制

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。

---

🎉 **恭喜！** 你现在拥有了一个完整的房源 RAG 知识库系统！

如果遇到任何问题，请查看上面的常见问题部分，或者检查 Docker 容器的日志输出。
