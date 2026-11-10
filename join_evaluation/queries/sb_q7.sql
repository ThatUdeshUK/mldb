-- id: sb_q7
-- source: evaluation/queries/sembench_movies/ipdb/q7.sql (D1:Q7, verbatim except for the LIMIT sweep)
-- description: k pairs of reviews of one movie with OPPOSITE sentiment (65,280 candidate pairs); the paper's D1:Q7 is k = 5000.
-- db: sembench_sf2000_icde.db
-- limits: 10, 30, 100, 300, 1000, 5000
-- true_cardinality: 65280, 256, 256
-- pair_columns: 1,2
-- ground_truth: opposite_sentiment
SELECT r1.id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM review AS r1
JOIN review AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania'
AND LLM <model> PROMPT 'These two movie reviews {{r1.reviewText}} and {{r2.reviewText}} express {opposite BOOL} sentiments, either positive or negative'
LIMIT <limit>;
