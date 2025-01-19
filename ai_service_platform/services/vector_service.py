from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import PyPDF2
import torch
import re
import jieba
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, VECTOR_DIM, VECTOR_MODEL_NAME

class VectorService:
    def __init__(self):
        self.model = SentenceTransformer(VECTOR_MODEL_NAME)
        self.connect_milvus()
        self.collection = self.ensure_collection()

    def connect_milvus(self):
        """Connect to Milvus server"""
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            if utility.has_collection(COLLECTION_NAME):
                print(f"\n删除已存在的集合: {COLLECTION_NAME}")
                utility.drop_collection(COLLECTION_NAME)
                
            print(f"\n创建新集合: {COLLECTION_NAME}")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
            ]
            schema = CollectionSchema(fields=fields, description="Document store")
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            
            # 创建索引
            print("\n创建索引...")
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print("索引创建完成")
            
            return collection
            
        except Exception as e:
            print(f"\n创建集合时出错: {str(e)}")
            raise

    def clean_text(self, text):
        """清理文本，移除特殊字符和多余的空白"""
        # 移除控制字符和特殊字符，但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uff60\x20-\x7e\n]', '', text)
        
        # 规范化中文标点
        text = text.replace('：', ':').replace('；', ';').replace('"', '"').replace('"', '"')
        
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除空行
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

    def split_text(self, text, max_length=800):
        """智能分割中文文本"""
        # 定义句子结束符模式
        sentence_pattern = (
            r'(?<=[。！？.!?])'  # 句号、感叹号、问号后面
            r'|(?<=[。！？.!?]["\'』」』》])'  # 带引号的句子结束
            r'|(?<=\n)'  # 换行符
        )
        
        # 分割句子
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果单个句子超过最大长度，使用结巴分词进一步分割
            if len(sentence) > max_length:
                words = list(jieba.cut(sentence))
                temp = ""
                for word in words:
                    if len(temp) + len(word) < max_length:
                        temp += word
                    else:
                        if temp:
                            chunks.append(temp)
                        temp = word
                if temp:
                    chunks.append(temp)
            else:
                # 尝试将句子添加到当前块
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 过滤太短的文本块
        return [chunk for chunk in chunks if len(chunk.strip()) > 30]

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text_chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        # 提取文本
                        text = page.extract_text()
                        if not text:
                            print(f"页面 {page_num} 未提取到文本")
                            continue
                            
                        # 清理文本
                        text = self.clean_text(text)
                        if not text:
                            print(f"页面 {page_num} 清理后无有效文本")
                            continue
                            
                        # 分割文本
                        chunks = self.split_text(text)
                        text_chunks.extend(chunks)
                        print(f"页面 {page_num} 提取了 {len(chunks)} 个文本块")
                        
                    except Exception as e:
                        print(f"处理页面 {page_num} 时出错: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"读取PDF文件时出错: {str(e)}")
            raise
            
        if not text_chunks:
            raise ValueError("未能从PDF中提取出有效文本")
            
        print(f"总共提取出 {len(text_chunks)} 个文本块")
        # 打印样本进行检查
        for i, chunk in enumerate(text_chunks[:2]):
            print(f"文本块样本 {i+1}: {chunk[:100]}...")
            
        return text_chunks

    def generate_embeddings(self, texts):
        """Generate embeddings for text chunks"""
        try:
            # 批量处理以避免内存问题
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, convert_to_tensor=True)
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
                print(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本块")
            
            return all_embeddings
            
        except Exception as e:
            print(f"生成向量嵌入时出错: {str(e)}")
            raise

    def insert_to_milvus(self, texts, embeddings):
        """Insert text and embeddings into Milvus"""
        try:
            print(f"\n准备插入数据到 Milvus:")
            print(f"文本数量: {len(texts)}")
            print(f"向量数量: {len(embeddings)}")
            if embeddings:
                print(f"向量维度: {len(embeddings[0])}")
            
            # 验证数据
            if not texts or not embeddings:
                raise ValueError("文本或向量为空")
            if len(texts) != len(embeddings):
                raise ValueError(f"文本数量 ({len(texts)}) 与向量数量 ({len(embeddings)}) 不匹配")
            
            # 获取集合
            print(f"\n当前集合信息:")
            print(f"名称: {self.collection.name}")
            print(f"行数: {self.collection.num_entities}")
            print(f"架构: {self.collection.schema}")
            
            # 准备数据
            entities = [
                texts,      # content field
                embeddings  # embedding field
            ]
            
            # 插入数据
            print("\n开始插入数据...")
            insert_result = self.collection.insert(entities)
            print(f"插入结果:")
            print(f"插入ID: {insert_result.primary_keys}")
            
            # 确保数据持久化
            print("\n刷新数据...")
            self.collection.flush()
            
            # 验证插入结果
            print("\n验证插入结果:")
            new_count = self.collection.num_entities
            print(f"当前集合实体数: {new_count}")
            
            if new_count == 0:
                raise ValueError("插入后集合仍为空")
            
            # 加载集合以供搜索
            print("\n加载集合...")
            self.collection.load()
            
            return new_count
            
        except Exception as e:
            print(f"\n插入文档时出错: {str(e)}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            raise

    def process_pdf(self, pdf_path):
        """Process PDF file and store in Milvus"""
        print(f"\n开始处理PDF文件: {pdf_path}")
        
        try:
            # Extract text from PDF
            text_chunks = self.extract_text_from_pdf(pdf_path)
            if not text_chunks:
                raise ValueError("未能从PDF中提取出有效文本")
                
            print(f"\n提取了 {len(text_chunks)} 个文本块")
            print("文本块样例:")
            for i, chunk in enumerate(text_chunks[:2]):
                print(f"[{i+1}] {chunk[:100]}...")
            
            print("\n开始生成向量嵌入...")
            # Generate embeddings
            embeddings = self.generate_embeddings(text_chunks)
            print(f"生成了 {len(embeddings)} 个向量")
            
            print("\n开始存储到向量数据库...")
            # Store in Milvus
            inserted_count = self.insert_to_milvus(text_chunks, embeddings)
            print(f"成功存储了 {inserted_count} 个文档")
            
            return inserted_count
            
        except Exception as e:
            print(f"\n处理PDF时出错: {str(e)}")
            raise
