import assert from "node:assert/strict";
import test from "node:test";

import {
  latestRatingTrainingTarget,
  latestTrainingTarget,
  nextRatingTrainingTarget,
  nextTrainingTarget,
  trainingIsDue,
} from "../lib/training-cadence";

test("training uses dense early milestones and sparse mature milestones", () => {
  assert.equal(nextTrainingTarget(null), 20);
  assert.equal(nextTrainingTarget(20), 40);
  assert.equal(nextTrainingTarget(39), 40);
  assert.equal(nextTrainingTarget(40), 60);
  assert.equal(nextTrainingTarget(70), 80);
  assert.equal(nextTrainingTarget(80), 100);
  assert.equal(nextTrainingTarget(100), 150);
  assert.equal(nextTrainingTarget(120), 150);
  assert.equal(nextTrainingTarget(150), 200);
});

test("training skips redundant milestones after a label-count leap", () => {
  assert.equal(latestTrainingTarget(19, null), 20);
  assert.equal(latestTrainingTarget(94, null), 80);
  assert.equal(latestTrainingTarget(94, 20), 80);
  assert.equal(latestTrainingTarget(149, 40), 100);
  assert.equal(latestTrainingTarget(260, 100), 250);
});

test("point ratings retrain every five early and ten after fifty", () => {
  assert.equal(nextRatingTrainingTarget(null), 5);
  assert.equal(nextRatingTrainingTarget(0), 5);
  assert.equal(nextRatingTrainingTarget(5), 10);
  assert.equal(nextRatingTrainingTarget(39), 40);
  assert.equal(nextRatingTrainingTarget(45), 50);
  assert.equal(nextRatingTrainingTarget(50), 60);
  assert.equal(nextRatingTrainingTarget(59), 60);
  assert.equal(nextRatingTrainingTarget(60), 70);
});

test("rating training skips redundant milestones after a label-count leap", () => {
  assert.equal(latestRatingTrainingTarget(4, null), 5);
  assert.equal(latestRatingTrainingTarget(34, null), 30);
  assert.equal(latestRatingTrainingTarget(34, 5), 30);
  assert.equal(latestRatingTrainingTarget(59, 10), 50);
  assert.equal(latestRatingTrainingTarget(86, 50), 80);
});

test("either feedback stream can make joint training due", () => {
  assert.equal(trainingIsDue(19, null, 4, null), false);
  assert.equal(trainingIsDue(20, null, 4, null), true);
  assert.equal(trainingIsDue(19, null, 5, null), true);
  assert.equal(trainingIsDue(39, 20, 9, 5), false);
  assert.equal(trainingIsDue(40, 20, 9, 5), true);
  assert.equal(trainingIsDue(39, 20, 10, 5), true);
  assert.equal(trainingIsDue(149, 120, 59, 50), false);
  assert.equal(trainingIsDue(150, 120, 59, 50), true);
  assert.equal(trainingIsDue(149, 120, 60, 50), true);
  assert.equal(trainingIsDue(95, 40, 0, null), true);
});

test("invalid prior model counts are rejected", () => {
  assert.throws(() => nextTrainingTarget(-1), RangeError);
  assert.throws(() => nextTrainingTarget(1.5), RangeError);
  assert.throws(() => nextRatingTrainingTarget(-1), RangeError);
  assert.throws(() => nextRatingTrainingTarget(1.5), RangeError);
  assert.throws(() => latestTrainingTarget(-1, null), RangeError);
  assert.throws(() => latestRatingTrainingTarget(-1, null), RangeError);
  assert.throws(() => trainingIsDue(-1, null, 0, null), RangeError);
  assert.throws(() => trainingIsDue(0, null, -1, null), RangeError);
});
