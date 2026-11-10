-- id: pc_q5
-- source: evaluation/queries/pc_parts/ipdb/q5.sql (D2:Q5, verbatim)
-- description: Compatible (CPU, motherboard) pairs (41 x 41 = 1,681 candidate pairs).
-- db: pc_part_icde.db
-- limit: none
-- pair_columns: 0,2
-- ground_truth: pc_compatible
SELECT CPUs.product_id, CPUS.product_name, Motherboards.product_id, Motherboards.product_name
FROM (Product AS p JOIN Category AS c1 ON p.category_id = c1.category_id AND c1.category_id = 4) AS Motherboards
JOIN (Product AS o JOIN Category AS c2 ON o.category_id = c2.category_id AND c2.category_id = 1) AS CPUs
ON LLM <model> PROMPT 'is CPU {{CPUs.PRODUCT_NAME}} {compatible BOOLEAN} with motherboard {{Motherboards.PRODUCT_NAME}}';
