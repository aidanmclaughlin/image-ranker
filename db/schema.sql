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
  discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, image_id),
  CHECK (matches = wins + losses)
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
  weights_json JSONB NOT NULL,
  artifact_blob_url TEXT,
  artifact_blob_path TEXT,
  metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  promoted BOOLEAN NOT NULL DEFAULT FALSE,
  promotion_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, comparison_cutoff)
);

ALTER TABLE model_runs
  ADD COLUMN IF NOT EXISTS promoted BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS promotion_reason TEXT;

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

CREATE INDEX IF NOT EXISTS idx_user_images_rank
  ON user_images(user_id, active, elo DESC, matches DESC);
CREATE INDEX IF NOT EXISTS idx_user_images_utility
  ON user_images(user_id, active, predicted_utility);
CREATE INDEX IF NOT EXISTS idx_pair_issuances_user_expiry
  ON pair_issuances(user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_pair_issuances_user_used
  ON pair_issuances(user_id, used_at)
  WHERE used_at IS NOT NULL;
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
CREATE INDEX IF NOT EXISTS idx_model_runs_user_created
  ON model_runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_runs_user_promoted
  ON model_runs(user_id, promoted, comparison_count DESC);
CREATE INDEX IF NOT EXISTS idx_worker_jobs_user_created
  ON worker_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_jobs_status
  ON worker_jobs(status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_jobs_single_active
  ON worker_jobs ((TRUE))
  WHERE status IN ('queued', 'running');
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_jobs_crawl_day
  ON worker_jobs(user_id, (input_json->>'run_day'))
  WHERE kind = 'crawl';
DROP INDEX IF EXISTS idx_worker_jobs_train_day;
CREATE UNIQUE INDEX IF NOT EXISTS idx_worker_jobs_train_cutoff_day
  ON worker_jobs(
    user_id,
    (input_json->>'comparison_cutoff'),
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

COMMIT;
