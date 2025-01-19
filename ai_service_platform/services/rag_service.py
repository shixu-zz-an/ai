from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re
import requests
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Generator
from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    QWEN_API_URL, QWEN_MODEL, VECTOR_MODEL_NAME
)

class RAGService:
    def __init__(self):
        """初始化 RAG 服务"""
        # 连接到 Milvus
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        # 加载向量模型
        self.model = SentenceTransformer(VECTOR_MODEL_NAME)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        
        # 确保集合存在
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保集合存在并具有正确的结构"""
        try:
            print(f"检查集合 {COLLECTION_NAME} 是否存在...")
            
            if utility.has_collection(COLLECTION_NAME):
                print(f"集合 {COLLECTION_NAME} 已存在")
                
                # 获取集合
                collection = Collection(COLLECTION_NAME)
                # 检查字段是否正确
                schema = collection.schema
                required_fields = {'id', 'content', 'embedding', 'source', 'page', 'chunk', 'total_pages'}
                existing_fields = {field.name for field in schema.fields}
                
                if not required_fields.issubset(existing_fields):
                    print(f"集合字段不完整，当前字段: {existing_fields}")
                    print(f"缺少字段: {required_fields - existing_fields}")
                    print("删除并重建集合...")
                    
                    utility.drop_collection(COLLECTION_NAME)
                    print(f"已删除集合 {COLLECTION_NAME}")
                else:
                    print("集合字段验证通过")
                    return
            
            # 创建集合
            print(f"创建新集合: {COLLECTION_NAME}")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="chunk", dtype=DataType.INT64),
                FieldSchema(name="total_pages", dtype=DataType.INT64)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"文档存储集合 {COLLECTION_NAME}",
                enable_dynamic_field=True
            )
            
            collection = Collection(
                name=COLLECTION_NAME,
                schema=schema,
                using='default',
                shards_num=2
            )
            
            # 创建索引
            print("创建向量索引...")
            index_params = {
                "metric_type": "L2",  # 使用 L2 距离
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            print("向量索引创建完成")
            
            print(f"集合 {COLLECTION_NAME} 创建成功")
            
        except Exception as e:
            print(f"确保集合存在时出错: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        Args:
            text: 原始文本
        Returns:
            处理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、：；（）【】《》""'']', '', text)
        return text.strip()

    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """将文本分割成适当大小的块
        Args:
            text: 要分割的文本
            max_length: 每个块的最大长度
        Returns:
            文本块列表
        """
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果段落太长，按句子分割
            if len(para) > max_length:
                sentences = re.split(r'([。！？!?])', para)
                i = 0
                while i < len(sentences):
                    if i + 1 < len(sentences):
                        # 将句子和标点符号组合在一起
                        sent = sentences[i] + sentences[i + 1]
                        i += 2
                    else:
                        sent = sentences[i]
                        i += 1
                        
                    sent = sent.strip()
                    if not sent:
                        continue
                        
                    # 如果当前块加上这个句子超过最大长度，保存当前块
                    if current_length + len(sent) > max_length and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.strip()) >= 50:  # 只保留有意义的块
                            chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sent)
                    current_length += len(sent)
            else:
                # 如果当前块加上这个段落超过最大长度，保存当前块
                if current_length + len(para) > max_length and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.strip()) >= 50:  # 只保留有意义的块
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(para)
                current_length += len(para)
        
        # 保存最后一个块
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= 50:  # 只保留有意义的块
                chunks.append(chunk_text)
        
        return chunks

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """处理 PDF 文件
        Args:
            file_path: PDF 文件路径
        Returns:
            处理结果，包含分块数量等信息
        """
        try:
            print(f"开始处理 PDF 文件: {file_path}")
            
            # 读取 PDF 文件
            pdf_reader = PdfReader(file_path)
            total_pages = len(pdf_reader.pages)
            print(f"PDF 总页数: {total_pages}")
            
            # 准备数据
            entities = []
            
            # 逐页处理
            for page_num, page in enumerate(pdf_reader.pages, 1):
                print(f"处理第 {page_num}/{total_pages} 页")
                
                # 提取文本
                text = page.extract_text()
                if not text.strip():
                    print(f"第 {page_num} 页没有文本内容")
                    continue
                
                # 分割文本
                chunks = self._split_text(text)
                print(f"第 {page_num} 页分割为 {len(chunks)} 个块")
                
                # 为每个块生成向量
                for chunk_num, chunk in enumerate(chunks, 1):
                    if not chunk.strip():
                        continue
                        
                    # 生成向量
                    embedding = self.model.encode(chunk, normalize_embeddings=True)  # 归一化向量
                    
                    # 准备实体数据
                    entity = {
                        'content': chunk,
                        'source': os.path.basename(file_path),
                        'page': page_num,
                        'chunk': chunk_num,
                        'total_pages': total_pages,
                        'embedding': embedding.tolist()
                    }
                    entities.append(entity)
                    
                    print(f"处理块 {chunk_num}，长度: {len(chunk)}")
            
            print(f"向量生成完成，准备插入 {len(entities)} 条记录")
            
            # 获取集合
            collection = Collection(COLLECTION_NAME)
            
            # 插入数据
            if entities:
                collection.insert(entities)
                print(f"成功插入 {len(entities)} 条记录")
                
                # 确保数据可用
                collection.flush()
                collection.load()
                
                return {
                    'status': 'success',
                    'pages': total_pages,
                    'chunks': len(entities)
                }
            else:
                return {
                    'status': 'error',
                    'message': '没有可插入的有效数据'
                }
                
        except Exception as e:
            print(f"处理 PDF 文件失败: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

   
    def query(self, query: str):
        """查询相关文档并生成回答"""
        try:
            print(f"开始处理查询: {query}")
            
            # 加载集合
            collection = Collection(COLLECTION_NAME)
            collection.load()
            print(f"成功加载集合: {COLLECTION_NAME}")
            
            # 检查集合是否为空
            try:
                row_count = collection.num_entities
                print(f"集合中的实体数量: {row_count}")
                
                if row_count == 0:
                    print("集合为空，没有可查询的文档")
                    yield f"data: {json.dumps({'sources': [], 'contexts': []}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'answer': '集合中还没有任何文档，请先上传一些 PDF 文件。'}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            except Exception as e:
                print(f"获取集合统计信息失败: {str(e)}")
                yield f"data: {json.dumps({'sources': [], 'contexts': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'error': f'获取集合信息失败: {str(e)}'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 预处理查询文本
            query = self._preprocess_text(query)
            
            # 生成查询向量
            query_vector = self.model.encode(query, normalize_embeddings=True)
            print(f"生成查询向量，维度: {len(query_vector)}")
            
            # 搜索相似文档
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=5,
                output_fields=["content", "source", "page", "chunk", "total_pages"]
            )
            
            # 提取相关文档内容
            contexts = []
            sources = []
            scores = []
            context_details = []
            
            for hits in results:
                for hit in hits:
                    content = hit.entity.get('content')
                    source = hit.entity.get('source')
                    page = hit.entity.get('page')
                    chunk = hit.entity.get('chunk')
                    total_pages = hit.entity.get('total_pages')
                    score = hit.score
                    
                    if content and source:
                        similarity = 1 / (1 + score)
                        if similarity > 0.3:
                            contexts.append(content)
                            source_info = f"{source} (第 {page}/{total_pages} 页)"
                            sources.append(source_info)
                            scores.append(similarity)
                            
                            context_details.append({
                                'content': content,
                                'source': source,
                                'page': page,
                                'total_pages': total_pages,
                                'chunk': chunk,
                                'similarity': f"{similarity:.4f}"
                            })

            if not contexts:
                yield f"data: {json.dumps({'sources': [], 'contexts': []}, ensure_ascii=False)}\n\n"
                message = "抱歉，我没有找到与您问题相关的内容。您可以：" + "\n" + "1. 尝试换个方式提问" + "\n" + "2. 确认相关文档是否已上传"
                yield f"data: {json.dumps({'answer': message}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 按相关性排序
            context_items = list(zip(contexts, sources, scores, context_details))
            context_items.sort(key=lambda x: float(x[2]), reverse=True)
            contexts, sources, scores, context_details = zip(*context_items)


            # 将来源和分数打包在一起
            source_items = []
            for source, score in zip(sources, scores):
                source_items.append({
                    'source': source,
                    'similarity': f"{score:.4f}"
                })
            
            # 先发送来源和上下文信息
            initial_data = {
                'sources': source_items,
                'contexts': context_details
            }
            yield f"data: {json.dumps(initial_data, ensure_ascii=False)}\n\n"

            
            # 构建提示
            prompt = f"""基于以下文档内容回答用户的问题。如果无法从文档中找到答案，请说明无法回答。
                如果文档内容与问题相关，请详细解释。如果不相关，请明确指出。

                文档内容：
                {' '.join(contexts)}

                用户问题：{query}

                请提供准确、详细的回答："""
            
            # 构建请求数据
            request_data = {
                "model": QWEN_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": True
            }
            
            response = requests.post(
                QWEN_API_URL,
                json=request_data,
                stream=True
            )
            response.raise_for_status()
            
            # 然后发送回答流
            for chunk in self.generate_answer(response):
                yield chunk
            
            # 最后发送完成标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"查询处理失败: {str(e)}")
            yield f"data: {json.dumps({'sources': [], 'contexts': []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'error': f'查询处理失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    def empty_generator(self, message: str = "抱歉，我没有找到相关的文档内容。"):
        """返回空结果的生成器
        Args:
            message: 要返回的消息
        """
        yield f"data: {json.dumps({'answer': message}, ensure_ascii=False)}\n\n"

    def generate_answer(self, response):
        """处理 Qwen API 的流式响应"""
        for line in response.iter_lines():
            if not line:
                continue

            try:
                # 解码行数据
                decoded_line = line.decode('utf-8')
                print(f"收到数据行: {decoded_line}")
                
                try:
                    data = json.loads(decoded_line)
                    
                    # 检查是否完成
                    if data.get('done', False):
                        yield "data: [DONE]\n\n"
                        break
                    
                    # 提取消息内容
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        if content:  # 确保内容不为空
                            response_data = {'answer': content}
                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                            
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"处理响应行时出错: {str(e)}")
                continue
