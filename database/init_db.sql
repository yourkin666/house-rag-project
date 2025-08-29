-- 房源 RAG 知识库数据库初始化脚本
-- 此脚本用于创建数据库表结构和必要的扩展

-- 启用 pgvector 扩展（支持向量操作）
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建房源表
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    location TEXT,
    price NUMERIC(15, 2), -- 支持最大 13 位整数，2 位小数，单位：万元
    description TEXT,
    -- 存储 Google text-embedding-004 模型生成的 768 维向量
    description_embedding VECTOR(768),
    -- 添加时间戳字段
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 为向量列创建索引以提高搜索性能
-- 使用 HNSW 索引，它对于向量相似性搜索非常高效
CREATE INDEX IF NOT EXISTS properties_embedding_hnsw_idx 
ON properties USING hnsw (description_embedding vector_cosine_ops);

-- 为常用查询字段创建索引
CREATE INDEX IF NOT EXISTS properties_location_idx ON properties(location);
CREATE INDEX IF NOT EXISTS properties_price_idx ON properties(price);
CREATE INDEX IF NOT EXISTS properties_created_at_idx ON properties(created_at);

-- 创建一个函数来自动更新 updated_at 字段
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 创建触发器，在更新记录时自动更新 updated_at 字段
DROP TRIGGER IF EXISTS update_properties_updated_at ON properties;
CREATE TRIGGER update_properties_updated_at
    BEFORE UPDATE ON properties
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 创建一个视图，用于查看已向量化和未向量化的房源统计
CREATE OR REPLACE VIEW properties_stats AS
SELECT 
    COUNT(*) as total_properties,
    COUNT(description_embedding) as embedded_properties,
    COUNT(*) - COUNT(description_embedding) as pending_embedding,
    ROUND(
        (COUNT(description_embedding)::NUMERIC / NULLIF(COUNT(*), 0)) * 100, 
        2
    ) as embedding_completion_percentage
FROM properties;

-- 输出初始化完成信息
DO $$
BEGIN
    RAISE NOTICE '✅ 数据库初始化完成！';
    RAISE NOTICE '📊 已创建表：properties';
    RAISE NOTICE '🔍 已创建索引：向量索引、位置索引、价格索引';
    RAISE NOTICE '📈 已创建视图：properties_stats (用于查看统计信息)';
    RAISE NOTICE '⚡ 已创建触发器：自动更新时间戳';
END $$;
