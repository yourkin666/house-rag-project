"""
异步优化的向量化处理和RAG核心逻辑模块
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document

from .embeddings import RAGService as BaseRAGService
from .database import db_manager

logger = logging.getLogger(__name__)


class AsyncRAGService(BaseRAGService):
    """异步优化的RAG服务类"""
    
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=4)  # 控制并发数
        logger.info("异步RAG服务初始化完成")
    
    def query_properties_async(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        异步优化的查询房源方法 - 阶段1：并行参数提取和搜索
        """
        try:
            # 1. 检查缓存（保持同步，因为很快）
            cache_key = hash(f"{question}_{max_results}")
            if cache_key in self._query_cache:
                logger.info("使用完整缓存结果，节省所有成本")
                self._query_stats['cache_hit_queries'] += 1
                cached_result = self._query_cache[cache_key]
                self._update_cost_stats(0, len(cached_result.get('retrieved_properties', [])))
                return cached_result
            
            # 2. 启动并行任务组
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 并行任务1：参数提取
                param_future = executor.submit(self._extract_search_parameters, question)
                
                # 并行任务2：计算动态K值（基于问题长度的快速估算）
                quick_k = self._quick_estimate_k(question)
                
                # 并行任务3：预热向量搜索（可以提前开始）
                vector_future = executor.submit(self._perform_vector_search, question, quick_k * 2)
                
                # 等待参数提取完成
                raw_search_params = param_future.result()
                search_params = self._clean_params_for_processing(raw_search_params)
                extraction_metadata = raw_search_params.get('_extraction_metadata', {})
                
                # 基于提取的参数重新计算精确的K值
                precise_k = self._calculate_dynamic_k(search_params, question)
                
                logger.info(f"用户查询分析: {search_params}")
                logger.info(f"并行优化 - 快速K: {quick_k}, 精确K: {precise_k}")
                
                # 获取向量搜索结果
                vector_results = vector_future.result()
                
                # 并行启动全文搜索和数据库查询准备
                fulltext_future = executor.submit(db_manager.fulltext_search, question, limit=precise_k * 2)
                
                # 等待全文搜索完成
                fulltext_results = fulltext_future.result()
            
            # 3. 执行RRF融合（相对较快，保持同步）
            if fulltext_results:
                hybrid_results = self.rrf_fusion.fuse_rankings(vector_results, fulltext_results, precise_k)
                logger.info(f"RRF融合后得到 {len(hybrid_results)} 个结果")
            else:
                logger.info("全文搜索无结果，使用纯向量搜索")
                hybrid_results = [type('HybridResult', (), {
                    'property_id': result[0], 
                    'final_score': result[1],
                    'vector_score': result[1],
                    'fulltext_score': 0.0
                })() for result in vector_results[:precise_k]]
            
            # 4. 并行处理：数据库批量查询 + LLM准备
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 并行任务1：批量获取房源详情
                property_ids = [result.property_id for result in hybrid_results[:max_results]]
                properties_future = executor.submit(db_manager.get_properties_by_ids, property_ids)
                
                # 并行任务2：准备LLM输入（可以在数据库查询的同时进行）
                # 先用property_ids构建临时上下文，等数据库结果完成后再完善
                temp_context_future = executor.submit(
                    self._prepare_temp_context, property_ids, search_params
                )
                
                # 等待数据库查询完成
                properties = properties_future.result()
                temp_context = temp_context_future.result()
            
            # 5. 构建最终上下文和生成回答
            docs = self._convert_properties_to_docs(properties, hybrid_results[:max_results])
            filtered_docs = self._rerank_and_filter(docs, search_params)
            context = self._format_docs_enhanced(filtered_docs[:max_results], search_params)
            
            # 6. 生成回答（LLM调用）
            answer = self._generate_answer_direct(context, question, str(search_params))
            
            # 7. 构建结果
            result = self._build_final_result(answer, filtered_docs[:max_results], search_params, extraction_metadata, precise_k)
            
            # 8. 缓存结果
            self._add_to_cache(self._query_cache, cache_key, result)
            
            # 9. 更新统计
            llm_calls = int(extraction_metadata.get('used_llm_fallback', False)) + 1
            self._update_cost_stats(llm_calls, len(result['retrieved_properties']))
            logger.info(f"异步查询完成！LLM调用次数: {llm_calls}")
            
            return result
            
        except Exception as e:
            logger.error(f"异步查询失败: {e}")
            # 回退到同步方法
            logger.info("回退到同步查询方法")
            return super().query_properties(question, max_results)
    
    def _quick_estimate_k(self, question: str) -> int:
        """快速估算K值，用于提前启动搜索"""
        base_k = 5
        
        # 基于问题长度的快速估算
        if len(question) > 50:
            base_k += 2
        if len(question) > 80:
            base_k += 1
            
        # 检测模糊查询关键词
        if any(word in question for word in ['推荐', '有什么', '看看']):
            base_k += 2
        
        # 检测复杂需求
        if len([w for w in ['价格', '位置', '房型', '学区'] if w in question]) >= 2:
            base_k += 1
        
        return min(base_k, 12)  # 上限控制
    
    def _prepare_temp_context(self, property_ids: List[int], search_params: Dict[str, Any]) -> str:
        """预先准备临时上下文信息"""
        # 这里可以做一些预处理工作，比如准备模板
        context_template = f"基于以下搜索条件：{search_params}，为用户找到了{len(property_ids)}个潜在房源..."
        return context_template
    
    def _convert_properties_to_docs(self, properties: List[Dict], hybrid_results: List[Any]) -> List[Document]:
        """将房源数据转换为Document对象，保持顺序"""
        # 创建property_id到property信息的映射
        prop_dict = {prop['id']: prop for prop in properties}
        
        docs = []
        for hybrid_result in hybrid_results:
            prop = prop_dict.get(hybrid_result.property_id)
            if prop:
                doc_content = f"房源：{prop['title']}。位于 {prop['location']}，价格 {prop['price']}万元。{prop['description']}"
                metadata = {
                    'property_id': prop['id'],
                    'title': prop['title'],
                    'location': prop['location'],
                    'price': prop['price'],
                    'hybrid_score': getattr(hybrid_result, 'final_score', 0.5),
                    'vector_score': getattr(hybrid_result, 'vector_score', 0.5),
                    'fulltext_score': getattr(hybrid_result, 'fulltext_score', 0.0)
                }
                docs.append(Document(page_content=doc_content, metadata=metadata))
        
        return docs
    
    def _build_final_result(self, answer: str, docs: List[Document], search_params: Dict[str, Any], 
                           extraction_metadata: Dict, dynamic_k: int) -> Dict[str, Any]:
        """构建最终结果"""
        retrieved_properties = []
        total_score = 0
        
        for doc in docs:
            metadata = doc.metadata
            match_score = self._calculate_match_score(doc, search_params)
            total_score += match_score
            
            property_info = {
                "id": metadata.get('property_id'),
                "title": metadata.get('title'),
                "location": metadata.get('location'),
                "price": metadata.get('price'),
                "match_score": round(match_score, 3),
                "match_percentage": int(match_score * 100),
                "match_reasons": self._get_match_reasons(doc, search_params)
            }
            retrieved_properties.append(property_info)
        
        avg_score = total_score / len(retrieved_properties) if retrieved_properties else 0
        search_quality = {
            "total_found": len(docs),
            "returned_count": len(retrieved_properties),
            "average_match_score": round(avg_score, 3),
            "search_quality_level": self._get_search_quality_level(avg_score),
            "used_cache": False,
            "dynamic_k_used": dynamic_k,
            "extraction_method": extraction_metadata.get('method', 'unknown'),
            "async_optimized": True  # 标记为异步优化版本
        }
        
        return {
            "answer": answer,
            "retrieved_properties": retrieved_properties,
            "query_analysis": search_params,
            "search_quality": search_quality
        }


# 异步优化的RAG服务实例
async_rag_service = AsyncRAGService()
