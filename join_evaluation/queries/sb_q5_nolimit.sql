-- id: sb_q5_nolimit
-- source: evaluation/queries/sembench_movies/ipdb/q5_nolimit_split.sql (D1:Q5 without LIMIT)
-- description: All pairs of reviews of one movie with the SAME sentiment (65,280 candidate pairs).
-- db: sembench_sf2000_icde.db
-- limit: none
-- optional: yes (expensive: run only when named in --queries)
-- pair_columns: 1,2
-- ground_truth: same_sentiment
SELECT r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM review  AS r1
JOIN review  AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND LLM <model> PROMPT 'These two movie reviews {{r1.reviewText}} and {{r2.reviewText}} express the {same_sentiment BOOL} - either both are positive or both are negative.';
