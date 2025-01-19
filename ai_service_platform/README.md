# AI Service Platform

这是一个基于Flask的AI服务平台，集成了RAG（检索增强生成）、文档处理和向量检索等功能。

## 功能特点

- PDF文档上传和向量化处理
- 基于Milvus的向量存储和检索
- 使用BGE-small-zh模型进行文本向量化
- 集成Qwen 2.5模型进行RAG检索
- 现代化的Web界面
- 可扩展的架构设计

## 系统要求

- Python 3.8+
- Milvus 2.3+
- Qwen 2.5服务
- GPU支持（推荐）

## 安装步骤

1. 克隆项目并安装依赖：
```bash
pip install -r requirements.txt
```

2. 确保Milvus服务正在运行（默认地址：localhost:19530）

3. 确保Qwen服务正在运行（默认地址：http://localhost:8000）

4. 配置服务：
   - 在 `config.py` 中根据需要调整Milvus和Qwen服务的配置

5. 启动应用：
```bash
python app.py
```

## 使用说明

1. 访问 http://localhost:5000 打开Web界面

2. 文档上传：
   - 点击"选择文件"上传PDF文档
   - 系统会自动处理文档并存储到向量库

3. 知识检索：
   - 在查询框输入问题
   - 系统会返回相关答案和参考文档片段

## 架构说明

- `app.py`: 主应用入口和路由控制
- `services/`:
  - `vector_service.py`: 处理文档向量化和Milvus存储
  - `rag_service.py`: 处理RAG查询和Qwen模型交互
- `templates/`: 前端界面模板
- `config.py`: 配置文件

## 扩展性

系统设计考虑了后续扩展性：

1. 支持添加新的AI模型和服务
2. 可以扩展文档处理类型
3. 可以集成更多的向量数据库
4. 支持添加Chatbot等新功能

## 注意事项

- 确保上传的PDF文件大小适中，避免处理超大文件
- 确保有足够的存储空间用于向量数据
- 建议在GPU环境下运行以获得更好的性能
