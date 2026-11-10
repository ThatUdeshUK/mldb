-- id: sb_q7_manual
-- source: evaluation/queries/sembench_movies/ipdb/q7_alt.sql (D1:Q7 decomposed by hand)
-- description: Hand-written decomposition of sb_q7_nolimit (extract each review's sentiment, join on <>).
--              Contains no LLM join, so it runs once in the 'manual' arm as a reference for the automatic rewrite.
-- db: sembench_sf2000_icde.db
-- limit: none
-- pair_columns: 1,2
-- ground_truth: opposite_sentiment
-- arms: manual
WITH sent_review AS (
SELECT *, LLM <model> PROMPT 'extract the {sentiment VARCHAR} of the movie
                               {{reviewText}}, either positive or negative'
FROM review
WHERE id = 'ant_man_and_the_wasp_quantumania'
)
SELECT r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM sent_review AS r1
JOIN sent_review AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.sentiment <> r2.sentiment;
