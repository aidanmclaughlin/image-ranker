#!/usr/bin/env node

import { randomBytes } from "node:crypto";

import { Client } from "pg";

import { safeErrorMessage } from "../lib/redaction";

type ComparisonRow = {
  left_elo: number;
  right_elo: number;
  delta: number;
  replayed: boolean;
};

type RatingRow = {
  point_rating: number;
  point_rated_at: Date;
  replayed: boolean;
};

async function main(): Promise<void> {
  const databaseUrl = (
    process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL
  )?.trim();
  if (!databaseUrl) {
    throw new Error("DATABASE_URL_UNPOOLED or DATABASE_URL is required");
  }

  const client = new Client({ connectionString: databaseUrl });
  await client.connect();
  try {
    await client.query("BEGIN");
    const nonce = randomBytes(16).toString("hex");
    const userId = `schema-smoke-${nonce}`;
    const tokenHash = randomBytes(32).toString("hex");
    const imageIds: number[] = [];

    for (const side of ["left", "right", "legacy-one", "legacy-two"]) {
      const sha = randomBytes(32).toString("hex");
      const inserted = await client.query<{ id: number }>(
        `INSERT INTO images(
           sha256, filename, original_blob_path, preview_blob_path,
           thumbnail_blob_path, width, height
         ) VALUES($1,$2,$3,$4,$5,2400,1600)
         RETURNING id`,
        [
          sha,
          `${side}.jpg`,
          `images/${sha}/original.jpg`,
          `images/${sha}/preview.webp`,
          `images/${sha}/thumb.webp`,
        ],
      );
      imageIds.push(inserted.rows[0].id);
    }

    for (const imageId of imageIds) {
      await client.query(
        "INSERT INTO user_images(user_id,image_id) VALUES($1,$2)",
        [userId, imageId],
      );
    }
    await client.query(
      `INSERT INTO pair_issuances(
         token_hash,user_id,left_id,right_id,expires_at
       ) VALUES($1,$2,$3,$4,now() + interval '1 hour')`,
      [tokenHash, userId, imageIds[0], imageIds[1]],
    );

    const parameters = [userId, imageIds[0], imageIds[1], imageIds[0], tokenHash];
    const first = await client.query<ComparisonRow>(
      "SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)",
      parameters,
    );
    const replay = await client.query<ComparisonRow>(
      "SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)",
      parameters,
    );
    const state = await client.query<{ comparisons: number; matches: number }>(
      `SELECT
         (SELECT COUNT(*)::integer FROM comparisons WHERE user_id=$1) AS comparisons,
         (SELECT SUM(matches)::integer FROM user_images WHERE user_id=$1) AS matches`,
      [userId],
    );

    if (first.rows[0]?.replayed || !replay.rows[0]?.replayed) {
      throw new Error("Comparison replay status is incorrect");
    }
    if (state.rows[0]?.comparisons !== 1 || state.rows[0]?.matches !== 2) {
      throw new Error("Comparison replay changed Elo state twice");
    }
    const job = await client.query<{ id: string }>(
      `INSERT INTO worker_jobs(user_id,kind,status,input_json,output_json)
       VALUES($1,'crawl','succeeded','{}'::jsonb,'{}'::jsonb)
       RETURNING id::text`,
      [userId],
    );
    const action = await client.query<{ id: string }>(
       `INSERT INTO crawl_bandit_actions(
         user_id,worker_job_id,action_index,arm,policy_version,propensity,
         status,proxy_reward,effective_reward,candidates_seen,candidates_eligible
       ) VALUES($1,$2,0,'test-arm','direct-rating-exp3-ix-v2',1.0,
                'observed',0.5,NULL,1,1)
       RETURNING id::text`,
      [userId, job.rows[0].id],
    );
    await client.query(
      `INSERT INTO crawl_bandit_discoveries(
         user_id,action_id,image_id,candidate_proxy_reward
       ) VALUES($1,$2,$3,0.5)`,
      [userId, action.rows[0].id, imageIds[0]],
    );
    await client.query("SAVEPOINT direct_discovery_single");
    try {
      await client.query(
        `INSERT INTO crawl_bandit_discoveries(
           user_id,action_id,image_id,candidate_proxy_reward
         ) VALUES($1,$2,$3,NULL)`,
        [userId, action.rows[0].id, imageIds[1]],
      );
      throw new Error("Direct source action accepted a second discovery");
    } catch (error) {
      const databaseError = error as { code?: string };
      await client.query("ROLLBACK TO SAVEPOINT direct_discovery_single");
      if (databaseError.code !== "23505") throw error;
    }
    const legacyAction = await client.query<{ id: string }>(
      `INSERT INTO crawl_bandit_actions(
         user_id,worker_job_id,action_index,arm,policy_version,propensity,
         status,proxy_reward,effective_reward,candidates_seen,candidates_eligible
       ) VALUES($1,$2,1,'legacy-arm','legacy-proxy-policy',1.0,
                'observed',0.5,0.5,2,2)
       RETURNING id::text`,
      [userId, job.rows[0].id],
    );
    await client.query(
      `INSERT INTO crawl_bandit_discoveries(
         user_id,action_id,image_id,candidate_proxy_reward
       ) VALUES($1,$2,$3,NULL),($1,$2,$4,NULL)`,
      [userId, legacyAction.rows[0].id, imageIds[2], imageIds[3]],
    );
    const ratingTokenHash = randomBytes(32).toString("hex");
    await client.query(
      `INSERT INTO rating_issuances(
         token_hash,user_id,image_id,expires_at
       ) VALUES($1,$2,$3,now() + interval '1 hour')`,
      [ratingTokenHash, userId, imageIds[0]],
    );
    const ratingParameters = [userId, imageIds[0], 5, ratingTokenHash];
    const firstRating = await client.query<RatingRow>(
      "SELECT * FROM record_user_rating($1,$2,$3,$4)",
      ratingParameters,
    );
    const replayedRating = await client.query<RatingRow>(
      "SELECT * FROM record_user_rating($1,$2,$3,$4)",
      ratingParameters,
    );
    const ratingState = await client.query<{
      ratings: number;
      point_rating: number;
      human_reward: number;
      effective_reward: number;
      human_matches: number;
    }>(
      `SELECT
         (SELECT COUNT(*)::integer FROM image_ratings WHERE user_id=$1) AS ratings,
         (SELECT point_rating FROM user_images
           WHERE user_id=$1 AND image_id=$2) AS point_rating,
         human_reward,
         effective_reward,
         human_matches
       FROM crawl_bandit_actions
       WHERE user_id=$1 AND id=$3`,
      [userId, imageIds[0], action.rows[0].id],
    );
    if (firstRating.rows[0]?.replayed || !replayedRating.rows[0]?.replayed) {
      throw new Error("Rating replay status is incorrect");
    }
    if (
      ratingState.rows[0]?.ratings !== 1 ||
      ratingState.rows[0]?.point_rating !== 5 ||
      ratingState.rows[0]?.human_reward !== 1 ||
      ratingState.rows[0]?.effective_reward !== 1 ||
      ratingState.rows[0]?.human_matches !== 1
    ) {
      throw new Error("Rating replay or crawler reward state is incorrect");
    }
    const legacyRatingTokenHash = randomBytes(32).toString("hex");
    await client.query(
      `INSERT INTO rating_issuances(
         token_hash,user_id,image_id,expires_at
       ) VALUES($1,$2,$3,now() + interval '1 hour')`,
      [legacyRatingTokenHash, userId, imageIds[2]],
    );
    await client.query(
      "SELECT * FROM record_user_rating($1,$2,$3,$4)",
      [userId, imageIds[2], 1, legacyRatingTokenHash],
    );
    const legacyReward = await client.query<{
      human_reward: number | null;
      effective_reward: number;
    }>(
      `SELECT human_reward,effective_reward
         FROM crawl_bandit_actions
        WHERE user_id=$1 AND id=$2`,
      [userId, legacyAction.rows[0].id],
    );
    if (
      legacyReward.rows[0]?.human_reward !== null ||
      legacyReward.rows[0]?.effective_reward !== 0.5
    ) {
      throw new Error("Direct ratings contaminated a legacy crawler policy");
    }

    const cutoffs = await client.query<{
      comparison_cutoff: string;
      rating_cutoff: string;
    }>(
      `SELECT
         (SELECT MAX(id)::text FROM comparisons WHERE user_id=$1)
           AS comparison_cutoff,
         (SELECT MAX(id)::text FROM image_ratings WHERE user_id=$1)
           AS rating_cutoff`,
      [userId],
    );
    const comparisonCutoff = cutoffs.rows[0].comparison_cutoff;
    const ratingCutoff = cutoffs.rows[0].rating_cutoff;
    await client.query(
      `INSERT INTO model_runs(
         user_id,encoder,comparison_cutoff,comparison_count,
         rating_cutoff,rating_count,feedback_count,weights_json
       ) VALUES
         ($1,'schema-smoke',$2,1,0,0,1,'{}'::jsonb),
         ($1,'schema-smoke',$2,1,$3,2,3,'{}'::jsonb)`,
      [userId, comparisonCutoff, ratingCutoff],
    );
    await client.query(
      `INSERT INTO worker_jobs(user_id,kind,status,input_json,output_json)
       VALUES
         ($1,'train','succeeded',
          jsonb_build_object('comparison_cutoff',$2::text,'rating_cutoff','0',
                             'run_day','2099-01-01'),'{}'::jsonb),
         ($1,'train','succeeded',
          jsonb_build_object('comparison_cutoff',$2::text,'rating_cutoff',$3::text,
                             'run_day','2099-01-01'),'{}'::jsonb)`,
      [userId, comparisonCutoff, ratingCutoff],
    );
    const trainIndex = await client.query<{ definition: string }>(
      `SELECT pg_get_indexdef(indexrelid) AS definition
         FROM pg_index
         JOIN pg_class ON pg_class.oid=indexrelid
        WHERE pg_class.relname='idx_worker_jobs_train_cutoff_day'`,
    );
    if (!trainIndex.rows[0]?.definition.includes("rating_cutoff")) {
      throw new Error("Training job uniqueness omits the rating cutoff");
    }
    await client.query(
      `INSERT INTO worker_jobs(user_id,kind,status,input_json,output_json)
       VALUES
         ($1,'crawl','succeeded',
          jsonb_build_object('rating_cutoff','100','run_day','2099-01-02'),
          '{}'::jsonb),
         ($1,'crawl','succeeded',
          jsonb_build_object('rating_cutoff','101','run_day','2099-01-02'),
          '{}'::jsonb),
         ($1,'crawl','failed',
          jsonb_build_object('rating_cutoff','100','run_day','2099-01-02'),
          '{}'::jsonb)`,
      [userId],
    );
    await client.query("SAVEPOINT duplicate_crawl_cutoff");
    try {
      await client.query(
        `INSERT INTO worker_jobs(user_id,kind,status,input_json,output_json)
         VALUES(
           $1,'crawl','succeeded',
           jsonb_build_object('rating_cutoff','100','run_day','2099-01-02'),
           '{}'::jsonb
         )`,
        [userId],
      );
      throw new Error("Duplicate crawl cutoff unexpectedly succeeded");
    } catch (error) {
      const databaseError = error as { code?: string };
      await client.query("ROLLBACK TO SAVEPOINT duplicate_crawl_cutoff");
      if (databaseError.code !== "23505") throw error;
    }
    const crawlIndex = await client.query<{ definition: string }>(
      `SELECT pg_get_indexdef(indexrelid) AS definition
         FROM pg_index
         JOIN pg_class ON pg_class.oid=indexrelid
        WHERE pg_class.relname='idx_worker_jobs_crawl_cutoff_day'`,
    );
    if (
      !crawlIndex.rows[0]?.definition.includes("rating_cutoff") ||
      !crawlIndex.rows[0]?.definition.includes("succeeded")
    ) {
      throw new Error("Crawl job uniqueness omits cutoff or retry state");
    }
    await client.query("SAVEPOINT immutable_rating");
    try {
      await client.query(
        "UPDATE image_ratings SET value=4 WHERE user_id=$1 AND image_id=$2",
        [userId, imageIds[0]],
      );
      throw new Error("Image rating update unexpectedly succeeded");
    } catch (error) {
      const databaseError = error as { code?: string };
      await client.query("ROLLBACK TO SAVEPOINT immutable_rating");
      if (databaseError.code !== "55000") throw error;
    }
    console.log("Hosted schema transaction smoke test passed");
  } finally {
    await client.query("ROLLBACK").catch(() => undefined);
    await client.end();
  }
}

main().catch((error: unknown) => {
  console.error(safeErrorMessage(error));
  process.exitCode = 1;
});
