# 房源 RAG 知识库项目 (重构版)

这是一个基于检索增强生成（RAG）技术的房源信息查询系统，使用 Google Gemini 作为大语言模型，PostgreSQL + pgvector 作为向量数据库。

## 🎯 重构说明

项目已经完成重构，采用了更规范的Python项目结构：
- **模块化设计**：将API、核心逻辑、配置分离
- **标准化结构**：遵循Python项目最佳实践
- **更好的维护性**：代码组织清晰，易于扩展和维护

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
├── src/                          # 源代码目录
│   └── house_rag/               # 主包
│       ├── __init__.py
│       ├── api/                 # API相关代码
│       │   ├── __init__.py
│       │   ├── app.py          # FastAPI应用主程序
│       │   └── models.py       # Pydantic数据模型
│       ├── core/               # 核心业务逻辑
│       │   ├── __init__.py
│       │   ├── config.py       # 配置管理
│       │   ├── database.py     # 数据库操作
│       │   └── embeddings.py   # RAG和向量化处理
│       └── scripts/            # 工具脚本
│           ├── __init__.py
│           └── ingest.py       # 数据向量化脚本
├── tests/                       # 测试文件
│   ├── __init__.py
│   └── test_append_api.py      # API测试脚本
├── database/                    # 数据库相关文件
│   ├── init_db.sql             # 数据库初始化脚本
│   └── sample_data.sql         # 示例数据
├── docs/                        # 文档目录
│   ├── README.md               # 详细文档 (原版)
│   ├── database_schema.md      # 数据库结构说明
│   └── 计划.md                 # 开发计划
├── app.py                      # 向后兼容的主入口文件
├── requirements.txt            # Python依赖包
├── Dockerfile                  # Docker镜像构建
├── docker-compose.yml         # Docker服务编排
└── .env                       # 环境变量配置
```

## 🚀 快速开始

### 方法一：使用重构后的模块化启动

```bash
# 启动API服务（推荐）
docker-compose exec app uvicorn src.house_rag.api.app:app --host 0.0.0.0 --port 8000 --reload

# 运行数据向量化脚本
docker-compose exec app python -m house_rag.scripts.ingest
```

### 方法二：使用向后兼容的启动方式

```bash
# 使用根目录的app.py启动（保持原有用法）
docker-compose exec app python app.py

# 或使用uvicorn
docker-compose exec app uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📦 Docker 部署

项目的Docker配置已更新，支持新的模块化结构：

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看应用日志
docker-compose logs -f app
```

## 🧪 开发和测试

### 运行测试
```bash
# 运行API测试
docker-compose exec app python tests/test_append_api.py

# 或从宿主机运行
python tests/test_append_api.py
```

### 数据库初始化
```bash
# 执行数据库初始化脚本
docker cp database/init_db.sql house_rag_db:/tmp/init_db.sql
docker-compose exec db psql -U houseuser -d house_knowledge_base -f /tmp/init_db.sql

# 插入示例数据
docker cp database/sample_data.sql house_rag_db:/tmp/sample_data.sql
docker-compose exec db psql -U houseuser -d house_knowledge_base -f /tmp/sample_data.sql
```

### 数据向量化处理
```bash
# 新的模块化方式（推荐）
docker-compose exec app python -m house_rag.scripts.ingest
```

## 🔧 配置说明

项目配置集中在 `src/house_rag/core/config.py` 中，通过环境变量进行管理：

```env
# .env 文件示例
GOOGLE_API_KEY="你的Google AI API密钥"
DB_HOST="localhost"
DB_PORT="5433"
DB_NAME="house_knowledge_base"
DB_USER="houseuser"
DB_PASSWORD="securepassword123"
DEBUG="false"
```

## 📖 API 使用

### 查询房源
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "我想找一个上海带游泳池的豪华别墅",
       "max_results": 3
     }'
```

### 添加房源
```bash
curl -X POST "http://localhost:8000/append" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "现代简约公寓",
       "location": "上海市徐汇区",
       "price": 680.0,
       "description": "位于徐汇区核心地段的现代简约公寓..."
     }'
```

### 获取统计信息
```bash
# 获取房源数量统计
curl -X GET "http://localhost:8000/properties/count"

# 健康检查
curl -X GET "http://localhost:8000/health"
```

## 🔄 重构带来的改进

1. **更清晰的代码组织**：按功能模块分离代码
2. **更好的可维护性**：每个模块职责单一
3. **更容易测试**：模块化设计便于单元测试
4. **更好的错误处理**：统一的错误处理机制
5. **更强的类型安全**：完整的类型注解
6. **更好的日志管理**：结构化的日志系统

## 📚 更多文档

- [详细使用说明](docs/README.md) - 完整的安装和使用指南
- [数据库结构说明](docs/database_schema.md) - 数据库表结构详解
- [开发计划](docs/计划.md) - 项目开发计划和进度

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。

---

🎉 **重构完成！** 项目现在拥有更好的结构和可维护性！
