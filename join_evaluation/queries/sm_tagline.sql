-- id: sm_tagline
-- source: NEW for this evaluation (D3 data). A join that is not attribute-based, the failure mode of paper
--         Sec. IV-B ("the relationship is not attribute-based"): which tagline belongs to which plot.
-- description: Match the taglines of 32 movies to their plots (32 x 32 = 1,024 candidate pairs, 32 true).
--              The movies are the first 32 (by id) with both a plot and a tagline; each contributes its
--              lexicographically smallest plot and tagline.
--              iPDB only decomposes an LLM predicate in a FILTER directly above the join (or above projections
--              over it), so this query is shaped for that, in every arm:
--                - a comma join with the LLM predicate in WHERE: with JOIN ... ON LLM over these derived tables,
--                  DuckDB keeps the predicate inside a nested-loop join condition;
--                - common_subplan disabled: the two sides share the movie_info scan, and DuckDB otherwise
--                  materializes it as a CTE between the filter and the cross product;
--                - literal movie ids rather than the same subquery on both sides, for the same reason.
-- db: benchmark_icde.db
-- limit: none
-- pair_columns: 0,1
-- ground_truth: same_movie
-- settings: SET disabled_optimizers = 'common_subplan';
SELECT p.movie_id AS plot_movie_id, t.movie_id AS tagline_movie_id
FROM (
    SELECT mi.movie_id, min(mi.info) AS plot
    FROM movie_info mi JOIN info_type it ON mi.info_type_id = it.id
    WHERE it.info = 'plot' AND mi.movie_id IN (1633389, 1633886, 1633937, 1633971, 1634712, 1635008, 1635380,
        1635656, 1636407, 1636761, 1637046, 1638273, 1638342, 1639178, 1639823, 1640322, 1640636, 1640674, 1640833,
        1641298, 1641694, 1642288, 1642666, 1642868, 1643297, 1643505, 1643870, 1645214, 1645902, 1646058, 1646065,
        1646295)
    GROUP BY mi.movie_id
) AS p, (
    SELECT mi.movie_id, min(mi.info) AS tagline
    FROM movie_info mi JOIN info_type it ON mi.info_type_id = it.id
    WHERE it.info = 'taglines' AND mi.movie_id IN (1633389, 1633886, 1633937, 1633971, 1634712, 1635008, 1635380,
        1635656, 1636407, 1636761, 1637046, 1638273, 1638342, 1639178, 1639823, 1640322, 1640636, 1640674, 1640833,
        1641298, 1641694, 1642288, 1642666, 1642868, 1643297, 1643505, 1643870, 1645214, 1645902, 1646058, 1646065,
        1646295)
    GROUP BY mi.movie_id
) AS t
WHERE LLM <model> PROMPT 'The tagline {{t.tagline}} {belongs BOOL} to the movie with the plot {{p.plot}}';
