-- id: sb_q5
-- source: evaluation/queries/sembench_movies/ipdb/q5.sql (D1:Q5, verbatim)
-- description: Ten pairs of reviews of one movie with the SAME sentiment (256 reviews -> 65,280 pairs).
-- db: sembench_sf2000_icde.db
-- limit: 10
-- pair_columns: 1,2
-- ground_truth: same_sentiment
SELECT r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM review  AS r1
JOIN review  AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND LLM <model> PROMPT 'These two movie reviews {{r1.reviewText}} and {{r2.reviewText}} express the {same_sentiment BOOL} - either both are positive or both are negative.'
LIMIT 10;
