BEGIN;

CREATE TABLE IF NOT EXISTS images (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  sha256 TEXT NOT NULL UNIQUE,
  filename TEXT NOT NULL,
  original_blob_path TEXT NOT NULL,
  preview_blob_path TEXT NOT NULL,
  thumbnail_blob_path TEXT NOT NULL,
  source_url TEXT,
  page_url TEXT,
  title TEXT,
  creator TEXT,
  license TEXT,
  width INTEGER NOT NULL CHECK (width > 0),
  height INTEGER NOT NULL CHECK (height > 0),
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_images (
  user_id TEXT NOT NULL,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  elo DOUBLE PRECISION NOT NULL DEFAULT 1500,
  matches INTEGER NOT NULL DEFAULT 0 CHECK (matches >= 0),
  wins INTEGER NOT NULL DEFAULT 0 CHECK (wins >= 0),
  losses INTEGER NOT NULL DEFAULT 0 CHECK (losses >= 0),
  active BOOLEAN NOT NULL DEFAULT TRUE,
  predicted_utility DOUBLE PRECISION,
  point_rating SMALLINT,
  point_rated_at TIMESTAMPTZ,
  discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, image_id),
  CHECK (matches = wins + losses),
  CONSTRAINT user_images_point_rating_valid CHECK (
    (point_rating IS NULL AND point_rated_at IS NULL)
    OR (
      point_rating IS NOT NULL
      AND point_rating BETWEEN 1 AND 5
      AND point_rated_at IS NOT NULL
    )
  )
);

ALTER TABLE user_images
  ADD COLUMN IF NOT EXISTS point_rating SMALLINT,
  ADD COLUMN IF NOT EXISTS point_rated_at TIMESTAMPTZ;
ALTER TABLE user_images
  DROP CONSTRAINT IF EXISTS user_images_point_rating_valid;
ALTER TABLE user_images
  ADD CONSTRAINT user_images_point_rating_valid CHECK (
    (point_rating IS NULL AND point_rated_at IS NULL)
    OR (
      point_rating IS NOT NULL
      AND point_rating BETWEEN 1 AND 5
      AND point_rated_at IS NOT NULL
    )
  );

CREATE TABLE IF NOT EXISTS pair_issuances (
  token_hash TEXT PRIMARY KEY CHECK (token_hash ~ '^[0-9a-f]{64}$'),
  user_id TEXT NOT NULL,
  left_id INTEGER NOT NULL,
  right_id INTEGER NOT NULL,
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  used_at TIMESTAMPTZ,
  CHECK (left_id <> right_id),
  CHECK (expires_at > issued_at),
  FOREIGN KEY (user_id, left_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE,
  FOREIGN KEY (user_id, right_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rating_issuances (
  token_hash TEXT PRIMARY KEY CHECK (token_hash ~ '^[0-9a-f]{64}$'),
  user_id TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  used_at TIMESTAMPTZ,
  CHECK (expires_at > issued_at),
  FOREIGN KEY (user_id, image_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS comparisons (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id TEXT NOT NULL,
  left_id INTEGER NOT NULL,
  right_id INTEGER NOT NULL,
  winner_id INTEGER NOT NULL,
  idempotency_key TEXT CHECK (
    idempotency_key IS NULL OR idempotency_key ~ '^[0-9a-f]{64}$'
  ),
  left_elo_before DOUBLE PRECISION NOT NULL,
  right_elo_before DOUBLE PRECISION NOT NULL,
  left_elo_after DOUBLE PRECISION,
  right_elo_after DOUBLE PRECISION,
  elo_delta DOUBLE PRECISION CHECK (elo_delta IS NULL OR elo_delta >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (left_id <> right_id),
  CHECK (winner_id = left_id OR winner_id = right_id),
  FOREIGN KEY (user_id, left_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE,
  FOREIGN KEY (user_id, right_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE,
  FOREIGN KEY (user_id, winner_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE
);

-- Keep this file safe to reapply while the hosted schema evolves. Historical
-- imported comparisons may not have an issuance token or stored outcome.
ALTER TABLE comparisons
  ADD COLUMN IF NOT EXISTS idempotency_key TEXT,
  ADD COLUMN IF NOT EXISTS left_elo_after DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS right_elo_after DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS elo_delta DOUBLE PRECISION;

-- Point ratings are immutable events. The current value on user_images is a
-- read-optimized projection written in the same transaction as this record.
CREATE TABLE IF NOT EXISTS image_ratings (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  value SMALLINT NOT NULL CHECK (value BETWEEN 1 AND 5),
  idempotency_key TEXT NOT NULL CHECK (idempotency_key ~ '^[0-9a-f]{64}$'),
  rated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, image_id),
  UNIQUE (user_id, idempotency_key),
  FOREIGN KEY (user_id, image_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE
);

CREATE OR REPLACE FUNCTION reject_image_rating_update()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  RAISE EXCEPTION 'Image ratings are immutable'
    USING ERRCODE = '55000';
END;
$$;

DROP TRIGGER IF EXISTS image_ratings_immutable ON image_ratings;
CREATE TRIGGER image_ratings_immutable
BEFORE UPDATE ON image_ratings
FOR EACH ROW EXECUTE FUNCTION reject_image_rating_update();

CREATE TABLE IF NOT EXISTS embeddings (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  encoder TEXT NOT NULL,
  vector BYTEA NOT NULL,
  dimensions INTEGER NOT NULL CHECK (dimensions > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (image_id, encoder)
);

CREATE TABLE IF NOT EXISTS model_runs (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id TEXT NOT NULL,
  encoder TEXT NOT NULL,
  comparison_cutoff BIGINT NOT NULL CHECK (comparison_cutoff >= 0),
  comparison_count INTEGER NOT NULL CHECK (comparison_count >= 0),
  rating_cutoff BIGINT NOT NULL DEFAULT 0 CHECK (rating_cutoff >= 0),
  rating_count INTEGER NOT NULL DEFAULT 0 CHECK (rating_count >= 0),
  feedback_count INTEGER NOT NULL CHECK (feedback_count >= 0),
  weights_json JSONB NOT NULL,
  artifact_blob_url TEXT,
  artifact_blob_path TEXT,
  metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  promoted BOOLEAN NOT NULL DEFAULT FALSE,
  promotion_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, comparison_cutoff, rating_cutoff)
);

ALTER TABLE model_runs
  ADD COLUMN IF NOT EXISTS promoted BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS promotion_reason TEXT,
  ADD COLUMN IF NOT EXISTS rating_cutoff BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS rating_count INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS feedback_count INTEGER NOT NULL DEFAULT 0;

UPDATE model_runs
   SET feedback_count = comparison_count + rating_count
 WHERE feedback_count <> comparison_count + rating_count;

ALTER TABLE model_runs
  DROP CONSTRAINT IF EXISTS model_runs_user_id_comparison_cutoff_key,
  DROP CONSTRAINT IF EXISTS model_runs_user_id_comparison_cutoff_rating_cutoff_key,
  DROP CONSTRAINT IF EXISTS model_runs_rating_cutoff_valid,
  DROP CONSTRAINT IF EXISTS model_runs_rating_count_valid,
  DROP CONSTRAINT IF EXISTS model_runs_feedback_count_valid;
ALTER TABLE model_runs
  ADD CONSTRAINT model_runs_user_id_comparison_cutoff_rating_cutoff_key
    UNIQUE (user_id, comparison_cutoff, rating_cutoff),
  ADD CONSTRAINT model_runs_rating_cutoff_valid CHECK (rating_cutoff >= 0),
  ADD CONSTRAINT model_runs_rating_count_valid CHECK (rating_count >= 0),
  ADD CONSTRAINT model_runs_feedback_count_valid
    CHECK (feedback_count = comparison_count + rating_count);

CREATE TABLE IF NOT EXISTS worker_jobs (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('train', 'crawl')),
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'skipped')),
  input_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  output_json JSONB,
  error TEXT,
  sandbox_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS crawl_bandit_actions (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id TEXT NOT NULL,
  worker_job_id BIGINT NOT NULL REFERENCES worker_jobs(id) ON DELETE CASCADE,
  action_index INTEGER NOT NULL CHECK (action_index >= 0),
  arm TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  propensity DOUBLE PRECISION NOT NULL
    CHECK (propensity > 0 AND propensity <= 1),
  model_run_id INTEGER REFERENCES model_runs(id) ON DELETE SET NULL,
  anchor_image_ids INTEGER[] NOT NULL DEFAULT '{}',
  context_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL
    CHECK (status IN ('chosen', 'observed', 'censored', 'failed')),
  proxy_reward DOUBLE PRECISION
    CHECK (proxy_reward BETWEEN 0 AND 1),
  human_reward DOUBLE PRECISION
    CHECK (human_reward BETWEEN 0 AND 1),
  human_matches INTEGER NOT NULL DEFAULT 0 CHECK (human_matches >= 0),
  effective_reward DOUBLE PRECISION
    CHECK (effective_reward BETWEEN 0 AND 1),
  candidates_seen INTEGER NOT NULL DEFAULT 0 CHECK (candidates_seen >= 0),
  candidates_eligible INTEGER NOT NULL DEFAULT 0 CHECK (candidates_eligible >= 0),
  chosen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  UNIQUE (worker_job_id, action_index),
  UNIQUE (user_id, id)
);

CREATE TABLE IF NOT EXISTS crawl_bandit_discoveries (
  user_id TEXT NOT NULL,
  action_id BIGINT NOT NULL,
  image_id INTEGER NOT NULL,
  candidate_proxy_reward DOUBLE PRECISION
    CHECK (candidate_proxy_reward BETWEEN 0 AND 1),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, image_id),
  FOREIGN KEY (user_id, action_id)
    REFERENCES crawl_bandit_actions(user_id, id) ON DELETE CASCADE,
  FOREIGN KEY (user_id, image_id)
    REFERENCES user_images(user_id, image_id) ON DELETE CASCADE
);

ALTER TABLE crawl_bandit_discoveries
  ALTER COLUMN candidate_proxy_reward DROP NOT NULL;

CREATE OR REPLACE FUNCTION enforce_direct_discovery_single()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  action_policy_version TEXT;
BEGIN
  SELECT action.policy_version INTO action_policy_version
    FROM crawl_bandit_actions AS action
   WHERE action.user_id = NEW.user_id
     AND action.id = NEW.action_id
   FOR UPDATE;
  IF action_policy_version = 'direct-rating-exp3-ix-v2'
     AND EXISTS (
       SELECT 1
         FROM crawl_bandit_discoveries AS discovery
        WHERE discovery.user_id = NEW.user_id
          AND discovery.action_id = NEW.action_id
     ) THEN
    RAISE EXCEPTION 'A direct-rating source action can import only one image'
      USING ERRCODE = '23505';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS crawl_bandit_direct_discovery_single
  ON crawl_bandit_discoveries;
CREATE TRIGGER crawl_bandit_direct_discovery_single
BEFORE INSERT ON crawl_bandit_discoveries
FOR EACH ROW EXECUTE FUNCTION enforce_direct_discovery_single();

CREATE INDEX IF NOT EXISTS idx_user_images_rank
  ON user_images(user_id, active, elo DESC, matches DESC);
CREATE INDEX IF NOT EXISTS idx_user_images_utility
  ON user_images(user_id, active, predicted_utility);
CREATE INDEX IF NOT EXISTS idx_pair_issuances_user_expiry
  ON pair_issuances(user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_pair_issuances_user_used
  ON pair_issuances(user_id, used_at)
  WHERE used_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rating_issuances_user_expiry
  ON rating_issuances(user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_rating_issuances_user_used
  ON rating_issuances(user_id, used_at)
  WHERE used_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rating_issuances_user_image_issued
  ON rating_issuances(user_id, image_id, issued_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_comparisons_user_idempotency
  ON comparisons(user_id, idempotency_key)
  WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_comparisons_user_created
  ON comparisons(user_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_comparisons_user_recent
  ON comparisons(user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_comparisons_user_pair
  ON comparisons(
    user_id,
    (LEAST(left_id, right_id)),
    (GREATEST(left_id, right_id))
  );
CREATE INDEX IF NOT EXISTS idx_image_ratings_user_rated
  ON image_ratings(user_id, rated_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_user_images_point_queue
  ON user_images(user_id, active, discovered_at DESC, image_id DESC)
  WHERE point_rating IS NULL;
CREATE INDEX IF NOT EXISTS idx_model_runs_user_created
  ON model_runs(user_id, created_at DESC);
DROP INDEX IF EXISTS idx_model_runs_user_promoted;
CREATE INDEX idx_model_runs_user_promoted
  ON model_runs(user_id, promoted, feedback_count DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_worker_jobs_user_created
  ON worker_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_jobs_status
  ON worker_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_crawl_bandit_history
  ON crawl_bandit_actions(user_id, id DESC)
  WHERE status = 'observed' AND effective_reward IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_crawl_bandit_discoveries_action
  ON crawl_bandit_discoveries(user_id, action_id);
DROP INDEX IF EXISTS idx_crawl_bandit_direct_discovery_action;
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_jobs_single_active
  ON worker_jobs ((TRUE))
  WHERE status IN ('queued', 'running');
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_jobs_crawl_day
  ON worker_jobs(user_id, (input_json->>'run_day'))
  WHERE kind = 'crawl';
DROP INDEX IF EXISTS idx_worker_jobs_train_day;
DROP INDEX IF EXISTS idx_worker_jobs_train_cutoff_day;
CREATE UNIQUE INDEX idx_worker_jobs_train_cutoff_day
  ON worker_jobs(
    user_id,
    (input_json->>'comparison_cutoff'),
    (COALESCE(input_json->>'rating_cutoff','0')),
    (input_json->>'run_day')
  )
  WHERE kind = 'train';

CREATE OR REPLACE FUNCTION lumen_adaptive_k(match_count INTEGER)
RETURNS DOUBLE PRECISION
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
  SELECT GREATEST(16.0, 48.0 / SQRT(1.0 + match_count / 20.0));
$$;

CREATE OR REPLACE FUNCTION record_user_comparison(
  comparison_user_id TEXT,
  comparison_left_id INTEGER,
  comparison_right_id INTEGER,
  comparison_winner_id INTEGER,
  comparison_idempotency_key TEXT
)
RETURNS TABLE (
  left_elo DOUBLE PRECISION,
  right_elo DOUBLE PRECISION,
  delta DOUBLE PRECISION,
  replayed BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
  left_image user_images%ROWTYPE;
  right_image user_images%ROWTYPE;
  issuance pair_issuances%ROWTYPE;
  prior_comparison comparisons%ROWTYPE;
  left_score DOUBLE PRECISION;
  expected_left DOUBLE PRECISION;
  k_factor DOUBLE PRECISION;
  rating_delta DOUBLE PRECISION;
BEGIN
  IF comparison_left_id = comparison_right_id
     OR comparison_winner_id NOT IN (comparison_left_id, comparison_right_id) THEN
    RAISE EXCEPTION 'Winner must be one of two distinct images'
      USING ERRCODE = '22023';
  END IF;
  IF comparison_idempotency_key IS NULL
     OR comparison_idempotency_key !~ '^[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'A valid comparison token is required'
      USING ERRCODE = '22023';
  END IF;

  -- The opaque token is stored only as a digest. Locking the issuance makes
  -- concurrent retries serialize before either can update Elo.
  SELECT issued.* INTO issuance
    FROM pair_issuances AS issued
   WHERE issued.token_hash = comparison_idempotency_key
     AND issued.user_id = comparison_user_id
     AND issued.left_id = comparison_left_id
     AND issued.right_id = comparison_right_id
   FOR UPDATE;
  IF issuance.token_hash IS NULL THEN
    RAISE EXCEPTION 'Comparison token is invalid for this pair'
      USING ERRCODE = '22023';
  END IF;

  SELECT comparison.* INTO prior_comparison
    FROM comparisons AS comparison
   WHERE comparison.user_id = comparison_user_id
     AND comparison.idempotency_key = comparison_idempotency_key;
  IF prior_comparison.id IS NOT NULL THEN
    IF prior_comparison.winner_id <> comparison_winner_id THEN
      RAISE EXCEPTION 'Comparison token was already used for another winner'
        USING ERRCODE = '22023';
    END IF;
    RETURN QUERY SELECT
      prior_comparison.left_elo_after,
      prior_comparison.right_elo_after,
      prior_comparison.elo_delta,
      TRUE;
    RETURN;
  END IF;

  IF issuance.expires_at <= NOW() THEN
    RAISE EXCEPTION 'Comparison token has expired'
      USING ERRCODE = '22023';
  END IF;

  -- Always acquire both contestant locks in image-id order so simultaneous
  -- comparisons cannot deadlock or apply an Elo update to stale ratings.
  PERFORM 1
    FROM user_images AS ui
    JOIN images AS image ON image.id = ui.image_id
   WHERE ui.user_id = comparison_user_id
     AND ui.image_id IN (comparison_left_id, comparison_right_id)
     AND ui.active
     AND image.active
   ORDER BY ui.image_id
   FOR UPDATE OF ui;

  SELECT ui.* INTO left_image
    FROM user_images AS ui
    JOIN images AS image ON image.id = ui.image_id
   WHERE ui.user_id = comparison_user_id
     AND ui.image_id = comparison_left_id
     AND ui.active
     AND image.active;
  SELECT ui.* INTO right_image
    FROM user_images AS ui
    JOIN images AS image ON image.id = ui.image_id
   WHERE ui.user_id = comparison_user_id
     AND ui.image_id = comparison_right_id
     AND ui.active
     AND image.active;

  IF left_image.image_id IS NULL OR right_image.image_id IS NULL THEN
    RAISE EXCEPTION 'Both images must exist in the user library'
      USING ERRCODE = '22023';
  END IF;

  left_score := CASE WHEN comparison_winner_id = comparison_left_id THEN 1.0 ELSE 0.0 END;
  expected_left := 1.0 / (1.0 + POWER(10.0, (right_image.elo - left_image.elo) / 400.0));
  k_factor := LEAST(lumen_adaptive_k(left_image.matches), lumen_adaptive_k(right_image.matches));
  rating_delta := k_factor * (left_score - expected_left);

  UPDATE user_images
     SET elo = left_image.elo + rating_delta,
         matches = matches + 1,
         wins = wins + CASE WHEN left_score = 1.0 THEN 1 ELSE 0 END,
         losses = losses + CASE WHEN left_score = 0.0 THEN 1 ELSE 0 END
   WHERE user_id = comparison_user_id AND image_id = comparison_left_id;
  UPDATE user_images
     SET elo = right_image.elo - rating_delta,
         matches = matches + 1,
         wins = wins + CASE WHEN left_score = 0.0 THEN 1 ELSE 0 END,
         losses = losses + CASE WHEN left_score = 1.0 THEN 1 ELSE 0 END
   WHERE user_id = comparison_user_id AND image_id = comparison_right_id;

  INSERT INTO comparisons (
    user_id, left_id, right_id, winner_id, idempotency_key,
    left_elo_before, right_elo_before, left_elo_after, right_elo_after,
    elo_delta
  ) VALUES (
    comparison_user_id, comparison_left_id, comparison_right_id,
    comparison_winner_id, comparison_idempotency_key,
    left_image.elo, right_image.elo,
    left_image.elo + rating_delta, right_image.elo - rating_delta,
    ABS(rating_delta)
  );

  UPDATE pair_issuances
     SET used_at = NOW()
   WHERE token_hash = comparison_idempotency_key;

  RETURN QUERY SELECT
    left_image.elo + rating_delta,
    right_image.elo - rating_delta,
    ABS(rating_delta),
    FALSE;
END;
$$;

CREATE OR REPLACE FUNCTION record_user_rating(
  rating_user_id TEXT,
  rating_image_id INTEGER,
  rating_value SMALLINT,
  rating_idempotency_key TEXT
)
RETURNS TABLE (
  point_rating SMALLINT,
  point_rated_at TIMESTAMPTZ,
  replayed BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
  issuance rating_issuances%ROWTYPE;
  rated_image user_images%ROWTYPE;
  prior_rating image_ratings%ROWTYPE;
  applied_at TIMESTAMPTZ;
BEGIN
  IF rating_image_id IS NULL
     OR rating_image_id <= 0
     OR rating_value IS NULL
     OR rating_value NOT BETWEEN 1 AND 5 THEN
    RAISE EXCEPTION 'Rating must be between 1 and 5 for a valid image'
      USING ERRCODE = '22023';
  END IF;
  IF rating_idempotency_key IS NULL
     OR rating_idempotency_key !~ '^[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'A valid rating token is required'
      USING ERRCODE = '22023';
  END IF;

  -- Serialize all retries for this opaque issuance before checking its event.
  SELECT issued.* INTO issuance
    FROM rating_issuances AS issued
   WHERE issued.token_hash = rating_idempotency_key
     AND issued.user_id = rating_user_id
     AND issued.image_id = rating_image_id
   FOR UPDATE;
  IF issuance.token_hash IS NULL THEN
    RAISE EXCEPTION 'Rating token is invalid for this image'
      USING ERRCODE = '22023';
  END IF;

  SELECT rating.* INTO prior_rating
    FROM image_ratings AS rating
   WHERE rating.user_id = rating_user_id
     AND rating.idempotency_key = rating_idempotency_key;
  IF prior_rating.id IS NOT NULL THEN
    IF prior_rating.image_id <> rating_image_id
       OR prior_rating.value <> rating_value THEN
      RAISE EXCEPTION 'Rating token was already used for another rating'
        USING ERRCODE = '22023';
    END IF;
    RETURN QUERY SELECT prior_rating.value, prior_rating.rated_at, TRUE;
    RETURN;
  END IF;

  IF issuance.expires_at <= NOW() THEN
    RAISE EXCEPTION 'Rating token has expired'
      USING ERRCODE = '22023';
  END IF;

  SELECT ui.* INTO rated_image
    FROM user_images AS ui
    JOIN images AS image ON image.id = ui.image_id
   WHERE ui.user_id = rating_user_id
     AND ui.image_id = rating_image_id
     AND ui.active
     AND image.active
   FOR UPDATE OF ui;
  IF rated_image.image_id IS NULL THEN
    RAISE EXCEPTION 'Image must exist in the user library'
      USING ERRCODE = '22023';
  END IF;
  IF rated_image.point_rating IS NOT NULL THEN
    RAISE EXCEPTION 'Image was already rated'
      USING ERRCODE = '22023';
  END IF;

  applied_at := NOW();
  INSERT INTO image_ratings (
    user_id, image_id, value, idempotency_key, rated_at
  ) VALUES (
    rating_user_id, rating_image_id, rating_value,
    rating_idempotency_key, applied_at
  );
  UPDATE user_images
     SET point_rating = rating_value,
         point_rated_at = applied_at
   WHERE user_id = rating_user_id AND image_id = rating_image_id;
  UPDATE rating_issuances
     SET used_at = applied_at
   WHERE token_hash = rating_idempotency_key;
  UPDATE crawl_bandit_actions AS action
     SET human_reward = (rating_value - 1)::DOUBLE PRECISION / 4.0,
         effective_reward = (rating_value - 1)::DOUBLE PRECISION / 4.0,
         human_matches = 1
    FROM crawl_bandit_discoveries AS discovery
   WHERE discovery.user_id = rating_user_id
     AND discovery.image_id = rating_image_id
     AND action.user_id = discovery.user_id
     AND action.id = discovery.action_id
     AND action.policy_version = 'direct-rating-exp3-ix-v2'
     AND action.status = 'observed'
     AND action.effective_reward IS NULL;

  RETURN QUERY SELECT rating_value, applied_at, FALSE;
END;
$$;

COMMIT;
