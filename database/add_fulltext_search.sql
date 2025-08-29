-- 混合搜索数据库迁移脚本
-- 为 properties 表添加全文搜索支持

-- 1. 添加 tsvector 列来存储全文搜索向量
ALTER TABLE properties 
ADD COLUMN search_vector tsvector;

-- 2. 创建函数来生成搜索向量
-- 这个函数会将 title, location, description 合并成一个可搜索的文档
CREATE OR REPLACE FUNCTION generate_search_vector(
    p_title TEXT,
    p_location TEXT,
    p_description TEXT
) RETURNS tsvector AS $$
BEGIN
    -- 使用中文分词配置 'simple'，因为 PostgreSQL 默认不包含中文分词
    -- 在生产环境中，建议安装 zhparser 扩展来更好地处理中文分词
    RETURN to_tsvector('simple', 
        COALESCE(p_title, '') || ' ' ||
        COALESCE(p_location, '') || ' ' ||
        COALESCE(p_description, '')
    );
END;
$$ LANGUAGE plpgsql;

-- 3. 为现有数据生成搜索向量
UPDATE properties 
SET search_vector = generate_search_vector(title, location, description)
WHERE search_vector IS NULL;

-- 4. 创建触发器函数，当相关字段更新时自动更新 search_vector
CREATE OR REPLACE FUNCTION update_search_vector() 
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector = generate_search_vector(NEW.title, NEW.location, NEW.description);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. 创建触发器
DROP TRIGGER IF EXISTS trigger_update_search_vector ON properties;
CREATE TRIGGER trigger_update_search_vector
    BEFORE INSERT OR UPDATE OF title, location, description
    ON properties
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- 6. 创建 GIN 索引来加速全文搜索
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_properties_search_vector 
ON properties USING GIN (search_vector);

-- 7. 添加一些有用的查询示例注释
/*
全文搜索查询示例：

-- 基本搜索
SELECT * FROM properties 
WHERE search_vector @@ to_tsquery('simple', '浦东 & 别墅');

-- 排序搜索（按相关度排序）
SELECT *, ts_rank(search_vector, to_tsquery('simple', '学区')) as relevance
FROM properties 
WHERE search_vector @@ to_tsquery('simple', '学区')
ORDER BY relevance DESC;

-- 前缀搜索
SELECT * FROM properties 
WHERE search_vector @@ to_tsquery('simple', '汤臣:*');

-- 短语搜索
SELECT * FROM properties 
WHERE search_vector @@ phraseto_tsquery('simple', '汤臣一品');
*/

-- 验证迁移结果
SELECT 
    'Migration completed successfully' as status,
    COUNT(*) as total_properties,
    COUNT(search_vector) as properties_with_search_vector
FROM properties;
