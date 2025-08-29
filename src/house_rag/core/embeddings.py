"""
向量化处理和RAG核心逻辑模块
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .config import config
from .database import db_manager

logger = logging.getLogger(__name__)


class RAGService:
    """RAG服务类，处理向量化和问答"""
    
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.rag_chain = None
        self._initialize()
    
    def _initialize(self) -> None:
        """初始化RAG组件"""
        try:
            # 初始化嵌入模型
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=config.GOOGLE_API_KEY
            )
            
            # 初始化LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.1,
                max_tokens=2048
            )
            
            # 初始化向量存储
            self.vector_store = PGVector(
                connection_string=config.pgvector_connection_string,
                embedding_function=self.embeddings,
                collection_name=config.COLLECTION_NAME,
                distance_strategy="cosine"
            )
            
            # 创建RAG链
            self._create_rag_chain()
            
            logger.info("RAG服务初始化完成")
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            raise
    
    def _create_rag_chain(self) -> None:
        """创建RAG处理链"""
        # 定义提示模板
        prompt_template = ChatPromptTemplate.from_template("""
你是一个专业的房地产顾问助手。请根据以下房源信息回答用户的问题。

房源信息：
{context}

用户问题：{question}

请提供专业、详细的回答，包括：
1. 直接回答用户的问题
2. 推荐最匹配的房源
3. 简要说明推荐理由
4. 如果有多个选择，请按匹配度排序

回答要求：
- 语言自然、友好
- 信息准确、具体
- 突出房源特色和优势
- 如果没有完全匹配的房源，也要提供相近的选择
""")
        
        # 创建检索器
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 定义文档格式化函数
        def format_docs(docs):
            formatted_docs = []
            for doc in docs:
                metadata = doc.metadata
                content = f"""
房源标题：{metadata.get('title', '未知')}
位置：{metadata.get('location', '未知')}
价格：{metadata.get('price', '面议')}万元
详细描述：{doc.page_content}
---
"""
                formatted_docs.append(content)
            return "\n".join(formatted_docs)
        
        # 创建RAG链
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成单个文本的向量"""
        try:
            embedding = self.embeddings.embed_query(text)
            logger.info(f"成功生成向量，维度: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"成功生成 {len(embeddings)} 个向量，维度: {len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"批量生成向量失败: {e}")
            raise
    
    def add_document_to_vectorstore(self, property_data: Dict[str, Any]) -> bool:
        """将房源数据添加到向量存储"""
        try:
            # 构建文档内容
            content = f"""房源：{property_data['title']}。位于 {property_data['location']}，价格 {property_data['price']}万元。{property_data['description']}"""
            
            # 创建文档对象
            document = Document(
                page_content=content,
                metadata={
                    "property_id": property_data['id'],
                    "title": property_data['title'],
                    "location": property_data['location'],
                    "price": property_data['price']
                }
            )
            
            # 添加到向量存储
            self.vector_store.add_documents([document])
            logger.info(f"成功将房源 {property_data['id']} 添加到向量存储")
            return True
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")
            raise
    
    def query_properties(self, question: str, max_results: int = 3) -> Dict[str, Any]:
        """
        查询房源并生成回答
        返回: {'answer': str, 'retrieved_properties': List[Dict]}
        """
        try:
            # 获取RAG回答
            answer = self.rag_chain.invoke(question)
            
            # 单独获取相关房源（用于返回结构化信息）
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": max_results}
            )
            retrieved_docs = retriever.invoke(question)
            
            # 格式化检索到的房源信息
            retrieved_properties = []
            for doc in retrieved_docs:
                metadata = doc.metadata
                retrieved_properties.append({
                    "id": metadata.get('property_id'),
                    "title": metadata.get('title'),
                    "location": metadata.get('location'),
                    "price": metadata.get('price')
                })
            
            return {
                "answer": answer,
                "retrieved_properties": retrieved_properties
            }
        except Exception as e:
            logger.error(f"查询房源失败: {e}")
            raise
    
    def vectorize_property(self, property_id: int, title: str, location: str, price: float, description: str) -> bool:
        """
        为单个房源生成向量并更新数据库
        返回: 是否成功
        """
        try:
            # 构建要向量化的文本
            text_to_vectorize = f"房源：{title}。位于 {location}，价格 {price}万元。{description}"
            
            # 生成向量
            embedding = self.generate_embedding(text_to_vectorize)
            
            # 更新数据库中的向量
            success = db_manager.update_property_embedding(property_id, embedding)
            
            if success:
                # 同时添加到向量存储以便搜索
                property_data = {
                    'id': property_id,
                    'title': title,
                    'location': location,
                    'price': price,
                    'description': description
                }
                self.add_document_to_vectorstore(property_data)
                logger.info(f"房源 {property_id} 向量化完成")
            
            return success
        except Exception as e:
            logger.error(f"房源向量化失败: {e}")
            raise


# 全局RAG服务实例
rag_service = RAGService()
