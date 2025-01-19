import os

# Milvus 配置
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'document_stor')
VECTOR_DIM = 768

# 文件上传配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
AUDIO_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')
ALLOWED_EXTENSIONS = {'pdf', 'wav', 'mp3'}

# 向量模型配置
VECTOR_MODEL_NAME = "shibing624/text2vec-base-chinese"

# Qwen API 配置
QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://127.0.0.1:11434/api/chat')
QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen2.5:7b')

# FunASR 配置
FUNASR_API_URL = os.getenv('FUNASR_API_URL', 'http://127.0.0.1:10096')

# Agent 配置
AGENT_FUNCTIONS = {
    'chat': {
        'name': '聊天',
        'description': '普通对话聊天'
    },
    'query': {
        'name': '文档检索',
        'description': '从文档中检索相关信息'
    },
    'speech': {
        'name': '语音处理',
        'description': '音频转写和总结'
    },
    'outbound': {
        'name': '外呼对话',
        'description': '模拟语音电话外呼'
    }
}
