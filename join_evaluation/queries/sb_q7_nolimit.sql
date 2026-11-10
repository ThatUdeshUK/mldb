-- id: sb_q7_nolimit
-- source: evaluation/queries/sembench_movies/ipdb/q7_split.sql (D1:Q7 without LIMIT)
-- description: All pairs of reviews of one movie with OPPOSITE sentiment (65,280 candidate pairs).
-- db: sembench_sf2000_icde.db
-- limit: none
-- pair_columns: 1,2
-- ground_truth: opposite_sentiment
SELECT r1.id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM review AS r1
JOIN review AS r2
ON LLM <model> PROMPT 'These two movie reviews {{r1.reviewText}} and {{r2.reviewText}} express {opposite BOOL} sentiments, either positive or negative' AND r1.id = r2.id
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND r2.id = 'ant_man_and_the_wasp_quantumania';
