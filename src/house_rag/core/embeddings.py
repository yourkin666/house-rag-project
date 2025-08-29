"""
向量化处理和RAG核心逻辑模块
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .config import config
from .database import db_manager

logger = logging.getLogger(__name__)


class HybridSearchResult:
    """混合搜索结果类"""
    
    def __init__(self, property_id: int, vector_score: float = 0.0, fulltext_score: float = 0.0, 
                 vector_rank: int = 0, fulltext_rank: int = 0, final_score: float = 0.0):
        self.property_id = property_id
        self.vector_score = vector_score
        self.fulltext_score = fulltext_score
        self.vector_rank = vector_rank
        self.fulltext_rank = fulltext_rank
        self.final_score = final_score
        
    def __repr__(self):
        return (f"HybridSearchResult(id={self.property_id}, "
                f"vector_score={self.vector_score:.3f}, "
                f"fulltext_score={self.fulltext_score:.3f}, "
                f"final_score={self.final_score:.3f})")


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) 算法实现
    
    RRF是一种无监督的排名融合方法，特别适合结合不同搜索系统的结果。
    它不需要预先知道各个搜索系统的准确性，能够自动平衡不同系统的贡献。
    
    公式: RRF_score(d) = Σ 1/(k + rank_i(d))
    其中 d 是文档，rank_i(d) 是文档 d 在第 i 个排名列表中的位置，k 是平滑参数
    """
    
    def __init__(self, k: int = 60):
        """
        初始化 RRF 融合器
        
        Args:
            k: RRF 平滑参数，默认60。较大的k值会减少高排名和低排名之间的差异
        """
        self.k = k
    
    def fuse_rankings(self, vector_results: List[Tuple[int, float]], 
                     fulltext_results: List[Tuple[int, float]], 
                     max_results: int = 50) -> List[HybridSearchResult]:
        """
        使用 RRF 算法融合向量搜索和全文搜索的结果
        
        Args:
            vector_results: 向量搜索结果 [(property_id, score), ...]
            fulltext_results: 全文搜索结果 [(property_id, score), ...]
            max_results: 返回的最大结果数
            
        Returns:
            List[HybridSearchResult]: 融合后的排序结果
        """
        # 收集所有唯一的property_id
        all_property_ids = set()
        vector_dict = {}
        fulltext_dict = {}
        
        # 处理向量搜索结果
        for rank, (prop_id, score) in enumerate(vector_results, 1):
            all_property_ids.add(prop_id)
            vector_dict[prop_id] = {'score': score, 'rank': rank}
        
        # 处理全文搜索结果
        for rank, (prop_id, score) in enumerate(fulltext_results, 1):
            all_property_ids.add(prop_id)
            fulltext_dict[prop_id] = {'score': score, 'rank': rank}
        
        # 计算融合分数
        hybrid_results = []
        for prop_id in all_property_ids:
            vector_info = vector_dict.get(prop_id, {'score': 0.0, 'rank': len(vector_results) + 1})
            fulltext_info = fulltext_dict.get(prop_id, {'score': 0.0, 'rank': len(fulltext_results) + 1})
            
            # RRF 分数计算
            rrf_score = 0.0
            if prop_id in vector_dict:
                rrf_score += 1.0 / (self.k + vector_info['rank'])
            if prop_id in fulltext_dict:
                rrf_score += 1.0 / (self.k + fulltext_info['rank'])
            
            hybrid_result = HybridSearchResult(
                property_id=prop_id,
                vector_score=vector_info['score'],
                fulltext_score=fulltext_info['score'],
                vector_rank=vector_info['rank'],
                fulltext_rank=fulltext_info['rank'],
                final_score=rrf_score
            )
            hybrid_results.append(hybrid_result)
        
        # 按融合分数排序
        hybrid_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return hybrid_results[:max_results]


class RAGService:
    """RAG服务类，处理向量化和问答"""
    
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.rag_chain = None
        self._query_cache = {}  # 简单的查询缓存
        
        # 成本控制和统计
        self._intent_cache = {}  # 意图分析缓存
        self._llm_call_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'keyword_fallbacks': 0,
            'hourly_calls': 0,
            'last_reset_time': None
        }
        
        # 混合搜索相关组件
        self.rrf_fusion = ReciprocalRankFusion(k=60)
        self.hybrid_search_enabled = True  # 控制是否启用混合搜索
        self.hybrid_search_stats = {
            'total_hybrid_searches': 0,
            'vector_only_fallbacks': 0,
            'fulltext_contributions': 0
        }
        
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
    
    def _assess_extraction_quality(self, params: Dict[str, Any], question: str) -> Dict[str, Any]:
        """评估规则提取结果的质量，决定是否需要LLM后备处理"""
        quality_score = 0
        max_score = 5
        extraction_info = {
            'quality_score': 0,
            'needs_llm_fallback': False,
            'reasons': [],
            'extracted_fields_count': 0
        }
        
        # 统计成功提取的字段数量
        extracted_fields = 0
        if params['price_range'] is not None:
            extracted_fields += 1
            quality_score += 1
        if params['location_keywords']:
            extracted_fields += 1
            quality_score += 1
        if params['property_type'] is not None:
            extracted_fields += 1
            quality_score += 1
        if params['area_preference'] is not None:
            extracted_fields += 1
            quality_score += 1
        if params['special_requirements']:
            extracted_fields += 1
            quality_score += 1
        
        extraction_info['extracted_fields_count'] = extracted_fields
        extraction_info['quality_score'] = quality_score
        
        # 决定是否需要LLM后备处理的逻辑
        question_length = len(question)
        
        # 情况1: 完全没有提取到任何信息
        if extracted_fields == 0:
            extraction_info['needs_llm_fallback'] = True
            extraction_info['reasons'].append("规则提取未找到任何参数")
        
        # 情况2: 问题很长但提取信息很少（可能包含复杂表达）
        elif question_length > 30 and extracted_fields <= 1:
            extraction_info['needs_llm_fallback'] = True
            extraction_info['reasons'].append("复杂查询但规则提取信息不足")
        
        # 情况3: 包含一些复杂的语言模式，规则可能无法处理
        complex_patterns = [
            '要么', '或者', '不过', '但是', '除了', '另外',
            '最好是', '希望', '比较', '相对', '大概', '差不多',
            '如果', '假如', '倘若'
        ]
        if any(pattern in question for pattern in complex_patterns):
            if extracted_fields <= 2:  # 复杂表达但信息提取少
                extraction_info['needs_llm_fallback'] = True
                extraction_info['reasons'].append("检测到复杂语言模式")
        
        # 情况4: 质量得分过低
        if quality_score < 2 and question_length > 15:
            extraction_info['needs_llm_fallback'] = True
            extraction_info['reasons'].append("整体提取质量偏低")
        
        logger.info(f"规则提取质量评估: 得分{quality_score}/{max_score}, "
                   f"提取字段{extracted_fields}个, 需要LLM后备: {extraction_info['needs_llm_fallback']}")
        
        return extraction_info

    def _extract_parameters_with_llm(self, question: str) -> Dict[str, Any]:
        """使用LLM进行结构化参数提取（后备方案）"""
        try:
            # 构建专门的提取指令
            extraction_prompt = f"""
你是一个专业的房产查询参数提取助手。请从用户的找房问题中提取出以下结构化信息，并严格按照JSON格式返回：

用户问题："{question}"

请提取以下信息（如果某项信息未提及或无法确定，请设为null）：

1. price_range: 价格区间，格式为[最小值, 最大值]，单位万元。如果只说了上限，最小值设为0
2. location_keywords: 地理位置关键词列表，提取所有提及的地名、区域等
3. property_type: 房屋类型，可选值："公寓"、"住宅"、"别墅"、"洋房"，只能选一个
4. area_preference: 面积偏好，提取数字，单位平方米
5. special_requirements: 特殊需求列表，如学区、地铁、停车、电梯等

请只返回JSON格式的结果，不要包含其他说明文字：
```json
{{
  "price_range": null 或 [数字1, 数字2],
  "location_keywords": [字符串列表],
  "property_type": null 或 "类型字符串", 
  "area_preference": null 或 数字,
  "special_requirements": [字符串列表]
}}
```"""

            # 调用LLM进行提取
            response = self.llm.invoke(extraction_prompt)
            response_text = response.content.strip()
            
            # 尝试从响应中提取JSON
            import json
            import re
            
            # 查找JSON块
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有代码块，尝试直接解析整个响应
                json_str = response_text
            
            # 解析JSON
            extracted_params = json.loads(json_str)
            
            # 数据清洗和验证
            cleaned_params = {
                'price_range': None,
                'location_keywords': [],
                'property_type': None,
                'area_preference': None,
                'special_requirements': []
            }
            
            # 处理价格范围
            if extracted_params.get('price_range'):
                price_range = extracted_params['price_range']
                if isinstance(price_range, list) and len(price_range) == 2:
                    cleaned_params['price_range'] = tuple(price_range)
            
            # 处理地理位置
            if extracted_params.get('location_keywords'):
                if isinstance(extracted_params['location_keywords'], list):
                    cleaned_params['location_keywords'] = extracted_params['location_keywords']
            
            # 处理房屋类型
            if extracted_params.get('property_type'):
                valid_types = ['公寓', '住宅', '别墅', '洋房']
                if extracted_params['property_type'] in valid_types:
                    cleaned_params['property_type'] = extracted_params['property_type']
            
            # 处理面积偏好
            if extracted_params.get('area_preference'):
                if isinstance(extracted_params['area_preference'], (int, float)):
                    cleaned_params['area_preference'] = int(extracted_params['area_preference'])
            
            # 处理特殊需求
            if extracted_params.get('special_requirements'):
                if isinstance(extracted_params['special_requirements'], list):
                    cleaned_params['special_requirements'] = extracted_params['special_requirements']
            
            logger.info(f"LLM提取成功: {cleaned_params}")
            return cleaned_params
            
        except Exception as e:
            logger.error(f"LLM参数提取失败: {e}")
            # 返回空参数作为降级方案
            return {
                'price_range': None,
                'location_keywords': [],
                'property_type': None,
                'area_preference': None,
                'special_requirements': []
            }

    def _extract_search_parameters_rule_based(self, question: str) -> Dict[str, Any]:
        """基于规则的参数提取（原有逻辑）"""
        params = {
            'price_range': None,
            'location_keywords': [],
            'property_type': None,
            'area_preference': None,
            'special_requirements': []
        }
        
        # 提取价格范围
        price_patterns = [
            r'(\d+)(?:万)?[-到](\d+)万',
            r'(\d+)-(\d+)万',
            r'(\d+)万以内',
            r'不超过(\d+)万',
            r'预算(\d+)万?左右',
            r'(\d+)万左右'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, question)
            if match:
                if len(match.groups()) == 2:
                    params['price_range'] = (int(match.group(1)), int(match.group(2)))
                else:
                    params['price_range'] = (0, int(match.group(1)))
                break
        
        # 提取地理位置关键词
        location_markers = ['区', '路', '街', '镇', '市', '县', '新区', '开发区', '附近', '周边']
        words = re.findall(r'[\u4e00-\u9fff]+', question)  # 提取中文词汇
        
        for word in words:
            if any(marker in word for marker in location_markers):
                params['location_keywords'].append(word)
        
        # 提取房屋类型
        if any(keyword in question for keyword in ['公寓', '住宅', '别墅', '洋房']):
            for prop_type in ['公寓', '住宅', '别墅', '洋房']:
                if prop_type in question:
                    params['property_type'] = prop_type
                    break
        
        # 提取面积偏好
        area_match = re.search(r'(\d+)(?:平|平方|㎡)', question)
        if area_match:
            params['area_preference'] = int(area_match.group(1))
        
        # 提取特殊需求
        special_keywords = ['学区', '地铁', '商圈', '医院', '公园', '停车', '电梯', '朝南']
        for keyword in special_keywords:
            if keyword in question:
                params['special_requirements'].append(keyword)
        
        return params

    def _extract_search_parameters(self, question: str) -> Dict[str, Any]:
        """
        混合模式参数提取：首先使用规则提取，必要时使用LLM后备
        """
        # 第一阶段：规则提取
        rule_based_params = self._extract_search_parameters_rule_based(question)
        
        # 第二阶段：质量评估
        quality_info = self._assess_extraction_quality(rule_based_params, question)
        
        # 第三阶段：决定是否需要LLM后备
        if quality_info['needs_llm_fallback']:
            logger.info(f"触发LLM后备提取，原因: {', '.join(quality_info['reasons'])}")
            
            # 使用LLM提取
            llm_params = self._extract_parameters_with_llm(question)
            
            # 合并规则提取和LLM提取结果
            merged_params = self._merge_extraction_results(rule_based_params, llm_params, question)
            
            # 添加提取元数据
            merged_params['_extraction_metadata'] = {
                'method': 'hybrid',
                'rule_quality': quality_info,
                'used_llm_fallback': True,
                'fallback_reasons': quality_info['reasons']
            }
            
            return merged_params
        
        else:
            logger.info("规则提取质量良好，使用规则提取结果")
            # 添加提取元数据
            rule_based_params['_extraction_metadata'] = {
                'method': 'rule_based_only',
                'rule_quality': quality_info,
                'used_llm_fallback': False
            }
            
            return rule_based_params

    def _merge_extraction_results(self, rule_params: Dict[str, Any], llm_params: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        智能合并规则提取和LLM提取的结果
        策略：优先使用规则结果（更精确），LLM结果作为补充
        """
        merged = {
            'price_range': None,
            'location_keywords': [],
            'property_type': None,
            'area_preference': None,
            'special_requirements': []
        }
        
        # 价格范围：优先使用规则结果，因为规则对数字处理更准确
        merged['price_range'] = rule_params['price_range'] or llm_params['price_range']
        
        # 地理位置：合并两个结果，去重
        all_locations = set(rule_params['location_keywords'] + llm_params['location_keywords'])
        merged['location_keywords'] = list(all_locations)
        
        # 房屋类型：优先使用规则结果
        merged['property_type'] = rule_params['property_type'] or llm_params['property_type']
        
        # 面积偏好：优先使用规则结果
        merged['area_preference'] = rule_params['area_preference'] or llm_params['area_preference']
        
        # 特殊需求：合并两个结果，去重
        all_requirements = set(rule_params['special_requirements'] + llm_params['special_requirements'])
        merged['special_requirements'] = list(all_requirements)
        
        logger.info(f"合并提取结果完成: 规则字段{self._count_extracted_fields(rule_params)}个, "
                   f"LLM字段{self._count_extracted_fields(llm_params)}个, "
                   f"最终字段{self._count_extracted_fields(merged)}个")
        
        return merged

    def _count_extracted_fields(self, params: Dict[str, Any]) -> int:
        """计算提取到的有效字段数量"""
        count = 0
        if params.get('price_range'):
            count += 1
        if params.get('location_keywords'):
            count += 1
        if params.get('property_type'):
            count += 1
        if params.get('area_preference'):
            count += 1
        if params.get('special_requirements'):
            count += 1
        return count

    def _clean_params_for_processing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """移除参数中的元数据，返回干净的处理参数"""
        clean_params = {
            'price_range': params.get('price_range'),
            'location_keywords': params.get('location_keywords', []),
            'property_type': params.get('property_type'),
            'area_preference': params.get('area_preference'),
            'special_requirements': params.get('special_requirements', [])
        }
        return clean_params
    
    def _calculate_dynamic_k(self, search_params: Dict[str, Any], question: str, base_k: int = 5, max_k: int = 12, min_k: int = 4) -> int:
        """
        根据查询复杂度动态调整检索数量 (优化版)
        主要依据提取出的结构化参数数量来评估复杂度，比单纯的字符串分析更准确
        """
        complexity_score = 0
        
        # 1. 基于提取出的参数数量（最可靠的指标）
        # 每个有效的搜索维度都增加复杂度
        if search_params.get('price_range'):
            complexity_score += 1
        if search_params.get('location_keywords'):
            complexity_score += len(search_params['location_keywords'])  # 多个地点增加更多复杂度
        if search_params.get('property_type'):
            complexity_score += 1
        if search_params.get('area_preference'):
            complexity_score += 1
        # 特殊需求越多，复杂度越高
        if search_params.get('special_requirements'):
            complexity_score += len(search_params['special_requirements'])
        
        # 2. 基于问题中的逻辑连接词作为补充（处理复杂逻辑关系）
        logical_keywords = ['并且', '同时', '或者', '要么', '另外', '而且', '以及']
        logical_complexity = sum(1 for keyword in logical_keywords if keyword in question)
        complexity_score += logical_complexity
        
        # 3. 基于问题长度作为微调（长问题通常包含更多细节）
        query_length = len(question)
        if query_length > 80:
            complexity_score += 2
        elif query_length > 50:
            complexity_score += 1
            
        # 4. 检测模糊查询（需要更多结果来满足用户期望）
        vague_indicators = ['推荐', '有什么', '看看', '找找', '合适的']
        if any(indicator in question for indicator in vague_indicators):
            complexity_score += 1
        
        # 动态调整k值，确保在合理范围内
        adjusted_k = min(base_k + complexity_score, max_k)
        final_k = max(adjusted_k, min_k)
        
        # 记录复杂度分析日志，便于调试和优化
        logger.debug(f"复杂度分析 - 参数维度: {self._count_extracted_fields(search_params)}, "
                    f"逻辑词: {logical_complexity}, 长度: {query_length}, "
                    f"最终复杂度: {complexity_score}, K值: {final_k}")
        
        return final_k
    
    def _get_adaptive_retriever_config(self, question: str, dynamic_k: int) -> dict:
        """根据查询类型返回最佳的检索器配置（成本优化版）"""
        
        # 首先检查缓存
        cache_key = f"intent_{hash(question)}"
        if config.ENABLE_INTENT_CACHE and cache_key in self._intent_cache:
            logger.debug(f"使用缓存的意图分析结果: {question[:30]}...")
            self._llm_call_stats['cache_hits'] += 1
            return self._build_strategy_from_intents(self._intent_cache[cache_key], dynamic_k)
        
        # 成本优化：混合策略 - 先判断是否需要LLM
        if self._should_use_llm_analysis(question):
            try:
                # 只对复杂查询使用LLM
                intents = self._classify_intent_with_llm(question)
                logger.info(f"LLM意图分析 - 查询: {question[:30]}..., 结果: {intents}")
                
                # 更新统计
                self._llm_call_stats['total_calls'] += 1
                self._llm_call_stats['hourly_calls'] += 1
                
                # 缓存结果
                if config.ENABLE_INTENT_CACHE:
                    self._intent_cache[cache_key] = intents
                
                return self._build_strategy_from_intents(intents, dynamic_k)
                
            except Exception as e:
                logger.warning(f"LLM意图分析失败，回退到关键词匹配: {e}")
                return self._get_keyword_based_config(question, dynamic_k)
        else:
            # 简单查询直接使用关键词匹配，节省成本
            logger.debug(f"简单查询使用关键词匹配: {question[:30]}...")
            return self._get_keyword_based_config(question, dynamic_k)
    
    def _should_use_llm_analysis(self, question: str) -> bool:
        """判断是否需要使用LLM进行意图分析（成本控制）"""
        
        # 检查并重置小时计数器
        self._reset_hourly_counter_if_needed()
        
        # 成本控制：检查LLM调用频率限制
        if self._llm_call_stats['hourly_calls'] >= config.MAX_LLM_CALLS_PER_HOUR:
            logger.warning(f"LLM调用已达小时限制({config.MAX_LLM_CALLS_PER_HOUR})，使用关键词匹配")
            self._llm_call_stats['keyword_fallbacks'] += 1
            return False
        
        # 简单查询条件：直接用关键词匹配就足够
        simple_conditions = [
            len(question) < 15,  # 很短的查询
            any(simple in question for simple in ['推荐', '有什么', '看看']),  # 纯探索性
            bool(re.search(r'^\w+房$', question)),  # 如"学区房"、"二手房"
        ]
        
        if any(simple_conditions):
            logger.debug(f"简单查询，直接使用关键词: {question[:30]}...")
            return False
            
        # 复杂查询条件：需要LLM深度理解
        complex_conditions = [
            '不要' in question or '别' in question,  # 包含否定词
            len(re.findall(r'[，,]', question)) >= 2,  # 多个条件用逗号分隔
            any(compound in question for compound in ['而且', '但是', '不过', '同时']),  # 复合逻辑
            len([w for w in ['价格', '位置', '房型', '面积', '学区', '地铁'] if w in question]) >= 2,  # 多维度需求
            '性价比' in question and any(special in question for special in ['学区', '地铁', '景观']),  # 明显的复合需求
        ]
        
        # 如果满足复杂条件，使用LLM
        if any(complex_conditions):
            logger.debug(f"检测到复杂查询，使用LLM分析: {question[:50]}...")
            return True
            
        # 中等复杂度查询：随机采样，平衡成本和效果
        import random
        use_llm = random.random() < config.LLM_SAMPLING_RATE
        if use_llm:
            logger.debug(f"中等复杂查询随机选择LLM分析(概率{config.LLM_SAMPLING_RATE}): {question[:50]}...")
        
        return use_llm
    
    def _reset_hourly_counter_if_needed(self):
        """重置小时计数器（如果需要）"""
        import datetime
        
        now = datetime.datetime.now()
        last_reset = self._llm_call_stats.get('last_reset_time')
        
        if last_reset is None or (now - last_reset).seconds >= 3600:  # 超过1小时
            self._llm_call_stats['hourly_calls'] = 0
            self._llm_call_stats['last_reset_time'] = now
            logger.info("重置LLM小时调用计数器")
    
    def get_cost_stats(self) -> dict:
        """获取成本统计信息"""
        stats = self._llm_call_stats.copy()
        stats['cache_size'] = len(self._intent_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(1, stats['total_calls']) * 100
        )
        return stats
    
    def log_cost_summary(self):
        """记录成本使用摘要"""
        stats = self.get_cost_stats()
        logger.info(f"LLM使用统计 - 总调用: {stats['total_calls']}, "
                   f"缓存命中率: {stats['cache_hit_rate']:.1f}%, "
                   f"本小时调用: {stats['hourly_calls']}, "
                   f"关键词回退: {stats['keyword_fallbacks']}")
    
    def _classify_intent_with_llm(self, question: str) -> dict:
        """使用LLM进行智能意图分类"""
        
        prompt = f"""
请作为一个专业的房地产咨询师，细致分析以下用户查询的真实意图。

用户查询："{question}"

请从以下维度进行深入分析（每个维度评分0-10，0=完全不相关，10=高度相关）：

## 核心需求维度：
1. **价格敏感度 (price_sensitive)**：
   - 关键词：便宜、经济、实惠、性价比、划算、优惠、预算有限、省钱
   - 考虑：用户是否明确关心成本控制和经济效益

2. **高端需求 (luxury)**：
   - 关键词：豪华、高端、别墅、顶级、奢华、精品、品质、尊贵、豪宅
   - 考虑：用户是否追求品质和档次，对价格不敏感

3. **位置明确度 (location_specific)**：
   - 关键词：具体区域、路名、街道、地铁站、商圈、附近、周边
   - 考虑：用户是否有明确的地理位置要求

4. **特殊需求 (special_needs)**：
   - 关键词：学区、地铁、停车、电梯、花园、景观、采光、朝向、交通
   - 考虑：用户是否有特定功能或配套设施的要求

5. **查询模糊度 (vague)**：
   - 关键词：房子、住房、房源、推荐、看看、有什么、随便
   - 考虑：用户需求是否宽泛、探索性的、缺乏明确方向

## 特殊语义分析：
6. **否定意图识别 (negations)**：
   - 否定词：不要、别、不需要、不想要、除了...之外、不考虑
   - 限制词：太...、过于...、过分...
   - 例如："不要太偏远" → ["偏远位置"]，"不考虑老房子" → ["老旧房屋"]

7. **复合意图 (compound)**：
   - 判断是否同时包含2个或以上高优先级需求
   - 例如："性价比高的学区房" = 价格敏感 + 特殊需求

## 语境理解要点：
- 注意同义词和近义词（如：豪宅=别墅，实惠=便宜）
- 识别隐含意图（如："刚需"通常暗示价格敏感）
- 分析语气和紧迫度
- 考虑否定词对整体意图的影响

请严格按照以下JSON格式输出分析结果：
{{
    "price_sensitive": 数值,
    "luxury": 数值,
    "location_specific": 数值,
    "special_needs": 数值,
    "vague": 数值,
    "negations": ["具体的否定内容"],
    "compound": true或false,
    "primary_intent": "主导意图名称",
    "confidence": 分析置信度(0.0-1.0),
    "reasoning": "简短的分析理由"
}}
"""
        
        response = self.llm.invoke(prompt)
        
        # 解析LLM响应
        try:
            # 尝试提取JSON部分
            import json
            import re
            
            # 提取JSON内容
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("无法从LLM响应中提取JSON")
                
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
            # 返回默认的中性分析结果
            return {
                "price_sensitive": 5,
                "luxury": 5,
                "location_specific": 5,
                "special_needs": 5,
                "vague": 5,
                "negations": [],
                "compound": False,
                "primary_intent": "default",
                "confidence": 0.3
            }
    
    def _build_strategy_from_intents(self, intents: dict, dynamic_k: int) -> dict:
        """根据意图分析结果构建检索策略（增强版）"""
        
        # 获取各维度得分
        price_score = intents.get("price_sensitive", 0)
        luxury_score = intents.get("luxury", 0)
        location_score = intents.get("location_specific", 0)
        special_score = intents.get("special_needs", 0)
        vague_score = intents.get("vague", 0)
        is_compound = intents.get("compound", False)
        confidence = intents.get("confidence", 0.5)
        negations = intents.get("negations", [])
        
        # 记录详细分析日志
        logger.info(f"意图分析详情 - 价格敏感: {price_score}, 高端: {luxury_score}, "
                   f"位置: {location_score}, 特殊需求: {special_score}, 模糊: {vague_score}")
        if negations:
            logger.info(f"检测到否定意图: {negations}")
        
        # 低置信度时使用保守策略
        if confidence < 0.6:
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k + 1,  # 低置信度时适当增加结果
                    "score_threshold": 0.65  # 稍微降低要求
                }
            }
        
        # 处理包含否定词的查询 - 需要更大的结果集来过滤
        has_negations = len(negations) > 0
        if has_negations:
            logger.info("检测到否定意图，增加结果数量以便后续过滤")
            k_adjustment = 3  # 有否定词时显著增加结果数
        else:
            k_adjustment = 0
        
        # 处理复合意图：识别同时存在的多种需求
        high_scores = sum(1 for score in [price_score, luxury_score, location_score, special_score] 
                         if score >= 7)
        
        # 特殊复合策略：价格敏感 + 特殊需求的组合
        if price_score >= 7 and special_score >= 7:
            logger.info("检测到价格敏感+特殊需求复合查询")
            return {
                "search_type": "similarity",
                "search_kwargs": {"k": dynamic_k + 4 + k_adjustment},  # 需要大量选择来平衡价格和需求
                "strategy_reason": "price_sensitive_with_special_needs"
            }
        
        # 特殊复合策略：位置 + 高端需求的组合
        if location_score >= 7 and luxury_score >= 7:
            logger.info("检测到位置+高端需求复合查询")
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k + 1 + k_adjustment,
                    "score_threshold": 0.75  # 高端+位置，要求较高精度
                },
                "strategy_reason": "location_luxury_compound"
            }
        
        # 通用复合意图处理
        if is_compound or high_scores >= 2:
            logger.info(f"检测到复合查询，高分维度数量: {high_scores}")
            return {
                "search_type": "similarity",
                "search_kwargs": {"k": dynamic_k + 3 + k_adjustment},  # 复合查询需要更多选择
                "strategy_reason": "compound_intent"
            }
        
        # 单一明确意图的处理 - 按优先级顺序
        if luxury_score >= 8:
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k + k_adjustment,
                    "score_threshold": 0.78  # 高端查询，高标准
                },
                "strategy_reason": "luxury_focused"
            }
        
        if special_score >= 8:
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k + 1 + k_adjustment,
                    "score_threshold": 0.74  # 特殊需求，较高精度
                },
                "strategy_reason": "special_needs_focused"
            }
        
        if location_score >= 8:
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k + k_adjustment,
                    "score_threshold": 0.68  # 位置查询，适中标准
                },
                "strategy_reason": "location_focused"
            }
        
        if price_score >= 8:
            return {
                "search_type": "similarity",
                "search_kwargs": {"k": dynamic_k + 3 + k_adjustment},  # 价格敏感，更多选择
                "strategy_reason": "price_sensitive_focused"
            }
        
        if vague_score >= 8:
            return {
                "search_type": "similarity",
                "search_kwargs": {"k": dynamic_k + 4 + k_adjustment},  # 模糊查询，大量选择
                "strategy_reason": "vague_exploration"
            }
        
        # 平衡策略（中等得分或无明确主导意图）
        return {
            "search_type": "similarity_score_threshold",
            "search_kwargs": {
                "k": dynamic_k + k_adjustment,
                "score_threshold": 0.70
            },
            "strategy_reason": "balanced_default"
        }
    
    def _get_keyword_based_config(self, question: str, dynamic_k: int) -> dict:
        """基于关键词的传统匹配方式（作为LLM的后备方案）"""
        
        # 价格敏感查询 - 用户关心性价比，需要更多选择
        price_sensitive_keywords = ['便宜', '经济', '实惠', '性价比', '划算', '优惠']
        if any(kw in question for kw in price_sensitive_keywords):
            return {
                "search_type": "similarity",
                "search_kwargs": {"k": dynamic_k + 2}  # 增加结果数量
            }
        
        # 高端精准查询 - 用户要求高，提高匹配标准
        luxury_keywords = ['豪华', '高端', '别墅', '顶级', '奢华', '精品']
        if any(kw in question for kw in luxury_keywords):
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k,
                    "score_threshold": 0.75  # 提高相似度要求
                }
            }
        
        # 区域性查询 - 用户明确指定位置，适中策略
        location_indicators = ['区', '路', '街', '镇', '市', '附近', '周边']
        has_location = any(indicator in question for indicator in location_indicators)
        if has_location:
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k,
                    "score_threshold": 0.68  # 适中的相似度要求
                }
            }
        
        # 特殊需求查询 - 用户有具体要求，需要精准匹配
        special_needs = ['学区', '地铁', '停车', '电梯', '花园', '江景', '朝南']
        if any(need in question for need in special_needs):
            return {
                "search_type": "similarity_score_threshold",
                "search_kwargs": {
                    "k": dynamic_k + 1,
                    "score_threshold": 0.72  # 较高的相似度要求
                }
            }
        
        # 模糊查询 - 用户需求不明确，降低要求增加选择
        vague_keywords = ['房子', '住房', '房源', '推荐', '有什么']
        if any(kw in question for kw in vague_keywords) and len(question) < 20:
            return {
                "search_type": "similarity",
                "search_kwargs": {"k": dynamic_k + 3}  # 大幅增加结果数量
            }
        
        # 默认策略 - 平衡精度和覆盖率
        return {
            "search_type": "similarity_score_threshold",
            "search_kwargs": {
                "k": dynamic_k,
                "score_threshold": 0.7  # 标准相似度要求
            }
        }
    
    def _create_rag_chain(self) -> None:
        """创建RAG处理链"""
        # 定义提示模板
        prompt_template = ChatPromptTemplate.from_template("""
你是一个专业的房地产顾问助手。请根据以下房源信息回答用户的问题。

房源信息：
{context}

用户问题：{question}

查询分析：{query_analysis}

请提供专业、详细的回答，包括：
1. 直接回答用户的问题
2. 推荐最匹配的房源（按相关度排序）
3. 简要说明推荐理由和匹配程度
4. 如果没有完全匹配的房源，提供相近的替代选择

回答要求：
- 语言自然、友好
- 信息准确、具体
- 突出房源特色和优势
- 考虑用户的具体需求（价格、位置、特殊要求等）
""")
        
        # 智能检索函数
        def smart_retrieve(question: str):
            return self._smart_retrieval(question)
        
        # 创建RAG链
        self.rag_chain = (
            {"context": smart_retrieve, "question": RunnablePassthrough(), 
             "query_analysis": lambda x: self._clean_params_for_processing(self._extract_search_parameters(x))}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def _hybrid_search_and_rerank(self, question: str, search_params: Dict[str, Any], dynamic_k: int) -> List[Document]:
        """
        执行混合搜索：结合向量搜索和全文搜索，使用RRF算法融合结果
        
        Args:
            question: 用户查询
            search_params: 提取的搜索参数
            dynamic_k: 动态检索数量
            
        Returns:
            List[Document]: 融合重排序后的文档列表
        """
        try:
            self.hybrid_search_stats['total_hybrid_searches'] += 1
            
            logger.info("开始执行混合搜索...")
            
            # 1. 执行向量搜索
            vector_results = self._perform_vector_search(question, dynamic_k * 2)  # 增加召回量以提高融合效果
            logger.info(f"向量搜索返回 {len(vector_results)} 个结果")
            
            # 2. 执行全文搜索
            fulltext_results = db_manager.fulltext_search(question, limit=dynamic_k * 2)
            logger.info(f"全文搜索返回 {len(fulltext_results)} 个结果")
            
            # 3. 使用RRF算法融合结果
            if not fulltext_results:
                # 如果全文搜索没有结果，回退到纯向量搜索
                logger.info("全文搜索无结果，回退到纯向量搜索")
                self.hybrid_search_stats['vector_only_fallbacks'] += 1
                return self._convert_vector_results_to_docs(vector_results[:dynamic_k], search_params)
            
            if not vector_results:
                # 如果向量搜索没有结果，使用全文搜索结果
                logger.info("向量搜索无结果，使用全文搜索结果")
                return self._convert_fulltext_results_to_docs(fulltext_results[:dynamic_k])
            
            # 执行RRF融合
            hybrid_results = self.rrf_fusion.fuse_rankings(vector_results, fulltext_results, dynamic_k)
            logger.info(f"RRF融合后得到 {len(hybrid_results)} 个结果")
            
            # 统计有全文搜索贡献的结果
            fulltext_contributed = sum(1 for r in hybrid_results[:dynamic_k] if r.fulltext_score > 0)
            self.hybrid_search_stats['fulltext_contributions'] += fulltext_contributed
            
            # 4. 转换为Document对象并应用重排序过滤
            fused_docs = self._convert_hybrid_results_to_docs(hybrid_results[:dynamic_k])
            filtered_docs = self._rerank_and_filter(fused_docs, search_params)
            
            logger.info(f"混合搜索最终返回 {len(filtered_docs)} 个文档")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"混合搜索执行失败: {e}")
            logger.info("回退到纯向量搜索")
            self.hybrid_search_stats['vector_only_fallbacks'] += 1
            
            # 回退到传统向量搜索
            retriever_config = self._get_adaptive_retriever_config(question, dynamic_k)
            retriever = self.vector_store.as_retriever(**retriever_config)
            retrieved_docs = retriever.invoke(question)
            return self._rerank_and_filter(retrieved_docs, search_params)
    
    def _perform_vector_search(self, question: str, k: int) -> List[Tuple[int, float]]:
        """
        执行向量搜索，返回property_id和相似度分数的列表
        """
        try:
            # 使用现有的向量存储进行搜索
            retriever_config = self._get_adaptive_retriever_config(question, k)
            retriever = self.vector_store.as_retriever(**retriever_config)
            retrieved_docs = retriever.invoke(question)
            
            # 提取property_id和分数
            results = []
            for doc in retrieved_docs:
                if 'property_id' in doc.metadata:
                    # 这里使用一个简单的相似度评分，实际可以从检索器获取更精确的分数
                    # 在实际实现中，可以考虑使用similarity_search_with_score方法
                    results.append((doc.metadata['property_id'], 1.0 / (len(results) + 1)))
            
            return results
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _convert_vector_results_to_docs(self, vector_results: List[Tuple[int, float]], 
                                       search_params: Dict[str, Any]) -> List[Document]:
        """将向量搜索结果转换为Document对象"""
        if not vector_results:
            return []
        
        property_ids = [result[0] for result in vector_results]
        properties = db_manager.get_properties_by_ids(property_ids)
        
        docs = []
        for prop in properties:
            doc_content = f"房源：{prop['title']}。位于 {prop['location']}，价格 {prop['price']}万元。{prop['description']}"
            metadata = {
                'property_id': prop['id'],
                'title': prop['title'],
                'location': prop['location'],
                'price': prop['price']
            }
            docs.append(Document(page_content=doc_content, metadata=metadata))
        
        return docs
    
    def _convert_fulltext_results_to_docs(self, fulltext_results: List[Tuple[int, float]]) -> List[Document]:
        """将全文搜索结果转换为Document对象"""
        if not fulltext_results:
            return []
        
        property_ids = [result[0] for result in fulltext_results]
        properties = db_manager.get_properties_by_ids(property_ids)
        
        docs = []
        for prop in properties:
            doc_content = f"房源：{prop['title']}。位于 {prop['location']}，价格 {prop['price']}万元。{prop['description']}"
            metadata = {
                'property_id': prop['id'],
                'title': prop['title'],
                'location': prop['location'],
                'price': prop['price']
            }
            docs.append(Document(page_content=doc_content, metadata=metadata))
        
        return docs
    
    def _convert_hybrid_results_to_docs(self, hybrid_results: List[HybridSearchResult]) -> List[Document]:
        """将混合搜索结果转换为Document对象"""
        if not hybrid_results:
            return []
        
        property_ids = [result.property_id for result in hybrid_results]
        properties = db_manager.get_properties_by_ids(property_ids)
        
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
                    'hybrid_score': hybrid_result.final_score,
                    'vector_score': hybrid_result.vector_score,
                    'fulltext_score': hybrid_result.fulltext_score
                }
                docs.append(Document(page_content=doc_content, metadata=metadata))
        
        return docs

    def _smart_retrieval(self, question: str) -> str:
        """智能检索和结果处理"""
        try:
            # 1. 查询缓存检查
            cache_key = hash(question)
            if cache_key in self._query_cache:
                logger.info("使用缓存结果")
                cached_result = self._query_cache[cache_key]
                return cached_result['formatted_context']
            
            # 2. 提取搜索参数
            raw_search_params = self._extract_search_parameters(question)
            search_params = self._clean_params_for_processing(raw_search_params)
            
            # 记录提取信息（包含元数据）
            if '_extraction_metadata' in raw_search_params:
                metadata = raw_search_params['_extraction_metadata']
                logger.info(f"参数提取方式: {metadata['method']}, 使用LLM后备: {metadata['used_llm_fallback']}")
            
            logger.info(f"清理后的搜索参数: {search_params}")
            
            # 3. 动态调整检索数量
            dynamic_k = self._calculate_dynamic_k(search_params, question)
            logger.info(f"动态K值: {dynamic_k}")
            
            # 4. 执行混合搜索（向量搜索 + 全文搜索）
            if self.hybrid_search_enabled:
                filtered_docs = self._hybrid_search_and_rerank(question, search_params, dynamic_k)
            else:
                # 回退到纯向量搜索
                retriever_config = self._get_adaptive_retriever_config(question, dynamic_k)
                logger.info(f"检索策略: {retriever_config}")
                
                retriever = self.vector_store.as_retriever(**retriever_config)
                retrieved_docs = retriever.invoke(question)
                
                # 5. 结果重排序和过滤
                filtered_docs = self._rerank_and_filter(retrieved_docs, search_params)
            
            # 6. 格式化结果
            formatted_context = self._format_docs_enhanced(filtered_docs, search_params)
            
            # 7. 缓存结果（限制缓存大小）
            if len(self._query_cache) > 100:
                # 简单的LRU清理：删除最旧的条目
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            
            self._query_cache[cache_key] = {
                'formatted_context': formatted_context,
                'search_params': search_params
            }
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"智能检索失败: {e}")
            # 降级到简单检索
            return self._fallback_retrieval(question)
    
    def _rerank_and_filter(self, docs: List[Document], search_params: Dict[str, Any]) -> List[Document]:
        """结果重排序和过滤"""
        if not docs:
            return docs
        
        scored_docs = []
        
        for doc in docs:
            score = 1.0  # 基础分数
            metadata = doc.metadata
            
            # 价格匹配加分
            if search_params.get('price_range') and metadata.get('price'):
                try:
                    price = float(metadata['price'])
                    min_price, max_price = search_params['price_range']
                    if min_price <= price <= max_price:
                        score += 0.3  # 价格完全匹配
                    elif abs(price - max_price) / max_price < 0.2:  # 价格接近
                        score += 0.1
                except (ValueError, TypeError):
                    pass
            
            # 位置匹配加分
            if search_params.get('location_keywords'):
                location = metadata.get('location', '').lower()
                for keyword in search_params['location_keywords']:
                    if keyword.lower() in location:
                        score += 0.2
                        break
            
            # 特殊需求匹配
            if search_params.get('special_requirements'):
                content = doc.page_content.lower()
                for requirement in search_params['special_requirements']:
                    if requirement.lower() in content:
                        score += 0.15
            
            # 房屋类型匹配
            if search_params.get('property_type'):
                if search_params['property_type'].lower() in doc.page_content.lower():
                    score += 0.25
            
            scored_docs.append((score, doc))
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 返回前N个最佳匹配（最多8个）
        return [doc for score, doc in scored_docs[:8]]
    
    def _format_docs_enhanced(self, docs: List[Document], search_params: Dict[str, Any]) -> str:
        """增强的文档格式化"""
        if not docs:
            return "抱歉，没有找到符合条件的房源。"
        
        formatted_docs = []
        
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            
            # 匹配度计算
            match_indicators = []
            if search_params.get('price_range') and metadata.get('price'):
                try:
                    price = float(metadata['price'])
                    min_price, max_price = search_params['price_range']
                    if min_price <= price <= max_price:
                        match_indicators.append("价格匹配")
                except (ValueError, TypeError):
                    pass
            
            if search_params.get('location_keywords'):
                location = metadata.get('location', '').lower()
                for keyword in search_params['location_keywords']:
                    if keyword.lower() in location:
                        match_indicators.append("位置匹配")
                        break
            
            match_info = f" (匹配: {', '.join(match_indicators)})" if match_indicators else ""
            
            content = f"""
【房源 {i}】{match_info}
标题：{metadata.get('title', '未知')}
位置：{metadata.get('location', '未知')}  
价格：{metadata.get('price', '面议')}万元
详细描述：{doc.page_content}
---
"""
            formatted_docs.append(content)
        
        return "\n".join(formatted_docs)
    
    def _fallback_retrieval(self, question: str) -> str:
        """降级检索策略"""
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            docs = retriever.invoke(question)
            return self._format_docs_simple(docs)
        except Exception as e:
            logger.error(f"降级检索也失败: {e}")
            return "抱歉，检索系统暂时不可用。"
    
    def _format_docs_simple(self, docs: List[Document]) -> str:
        """简单的文档格式化"""
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
    
    def query_properties(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        智能查询房源并生成回答
        返回: {'answer': str, 'retrieved_properties': List[Dict], 'query_analysis': Dict, 'search_quality': Dict}
        """
        try:
            # 提取查询参数（包含元数据用于分析）
            raw_search_params = self._extract_search_parameters(question)
            search_params = self._clean_params_for_processing(raw_search_params)
            extraction_metadata = raw_search_params.get('_extraction_metadata', {})
            
            logger.info(f"用户查询分析: {search_params}")
            if extraction_metadata:
                logger.info(f"参数提取详情: {extraction_metadata}")
            
            # 获取RAG回答
            answer = self.rag_chain.invoke(question)
            
            # 获取检索结果用于详细分析
            cache_key = hash(question)
            if cache_key in self._query_cache:
                cached_data = self._query_cache[cache_key]
                # 注意：缓存中的search_params可能是旧格式，保持兼容
                cached_search_params = cached_data.get('search_params', {})
                if '_extraction_metadata' not in cached_search_params:
                    # 如果缓存的是清理后的参数，直接使用
                    search_params = cached_search_params
            
            # 重新检索以获取详细信息
            dynamic_k = self._calculate_dynamic_k(search_params, question)
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": dynamic_k,
                    "score_threshold": 0.6  # 稍微放宽阈值
                }
            )
            
            retrieved_docs = retriever.invoke(question)
            filtered_docs = self._rerank_and_filter(retrieved_docs, search_params)
            
            # 格式化检索到的房源信息
            retrieved_properties = []
            total_score = 0
            
            for doc in filtered_docs[:max_results]:
                metadata = doc.metadata
                
                # 计算匹配分数
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
            
            # 搜索质量分析
            avg_score = total_score / len(retrieved_properties) if retrieved_properties else 0
            search_quality = {
                "total_found": len(retrieved_docs),
                "returned_count": len(retrieved_properties),
                "average_match_score": round(avg_score, 3),
                "search_quality_level": self._get_search_quality_level(avg_score),
                "used_cache": cache_key in self._query_cache,
                "dynamic_k_used": dynamic_k
            }
            
            return {
                "answer": answer,
                "retrieved_properties": retrieved_properties,
                "query_analysis": search_params,
                "search_quality": search_quality
            }
            
        except Exception as e:
            logger.error(f"查询房源失败: {e}")
            # 返回错误但不抛出异常，提供降级服务
            return {
                "answer": "抱歉，搜索服务暂时不可用，请稍后重试。",
                "retrieved_properties": [],
                "query_analysis": {},
                "search_quality": {"error": str(e)}
            }
    
    def _calculate_match_score(self, doc: Document, search_params: Dict[str, Any]) -> float:
        """计算详细的匹配分数"""
        base_score = 0.5  # 基础分数
        metadata = doc.metadata
        
        # 价格匹配 (权重: 30%)
        if search_params.get('price_range') and metadata.get('price'):
            try:
                price = float(metadata['price'])
                min_price, max_price = search_params['price_range']
                if min_price <= price <= max_price:
                    base_score += 0.3
                elif abs(price - max_price) / max_price < 0.3:
                    base_score += 0.15
            except (ValueError, TypeError):
                pass
        
        # 位置匹配 (权重: 25%)
        if search_params.get('location_keywords'):
            location = metadata.get('location', '').lower()
            for keyword in search_params['location_keywords']:
                if keyword.lower() in location:
                    base_score += 0.25
                    break
        
        # 房屋类型匹配 (权重: 20%)
        if search_params.get('property_type'):
            if search_params['property_type'].lower() in doc.page_content.lower():
                base_score += 0.2
        
        # 特殊需求匹配 (权重: 15%)
        if search_params.get('special_requirements'):
            content = doc.page_content.lower()
            match_count = sum(1 for req in search_params['special_requirements'] 
                            if req.lower() in content)
            if match_count > 0:
                base_score += min(0.15 * match_count / len(search_params['special_requirements']), 0.15)
        
        # 面积匹配 (权重: 10%)
        if search_params.get('area_preference'):
            content = doc.page_content
            area_match = re.search(r'(\d+)(?:平|平方|㎡)', content)
            if area_match:
                actual_area = int(area_match.group(1))
                preferred_area = search_params['area_preference']
                if abs(actual_area - preferred_area) / preferred_area < 0.2:
                    base_score += 0.1
        
        return min(base_score, 1.0)  # 确保分数不超过1.0
    
    def _get_match_reasons(self, doc: Document, search_params: Dict[str, Any]) -> List[str]:
        """获取匹配原因说明"""
        reasons = []
        metadata = doc.metadata
        
        if search_params.get('price_range') and metadata.get('price'):
            try:
                price = float(metadata['price'])
                min_price, max_price = search_params['price_range']
                if min_price <= price <= max_price:
                    reasons.append(f"价格符合预算 {min_price}-{max_price}万")
            except (ValueError, TypeError):
                pass
        
        if search_params.get('location_keywords'):
            location = metadata.get('location', '').lower()
            for keyword in search_params['location_keywords']:
                if keyword.lower() in location:
                    reasons.append(f"位置匹配 {keyword}")
        
        if search_params.get('property_type'):
            if search_params['property_type'].lower() in doc.page_content.lower():
                reasons.append(f"房屋类型匹配 {search_params['property_type']}")
        
        if search_params.get('special_requirements'):
            content = doc.page_content.lower()
            for req in search_params['special_requirements']:
                if req.lower() in content:
                    reasons.append(f"满足特殊需求: {req}")
        
        return reasons
    
    def _get_search_quality_level(self, avg_score: float) -> str:
        """根据平均分数确定搜索质量等级"""
        if avg_score >= 0.8:
            return "优秀"
        elif avg_score >= 0.6:
            return "良好"
        elif avg_score >= 0.4:
            return "一般"
        else:
            return "较差"
    
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
