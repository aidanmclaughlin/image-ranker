import assert from "node:assert/strict";
import test from "node:test";

import {
  latestTrainingTarget,
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

test("training becomes due exactly at each next milestone", () => {
  assert.equal(trainingIsDue(19, null), false);
  assert.equal(trainingIsDue(20, null), true);
  assert.equal(trainingIsDue(39, 20), false);
  assert.equal(trainingIsDue(40, 20), true);
  assert.equal(trainingIsDue(59, 40), false);
  assert.equal(trainingIsDue(60, 40), true);
  assert.equal(trainingIsDue(79, 70), false);
  assert.equal(trainingIsDue(80, 70), true);
  assert.equal(trainingIsDue(99, 80), false);
  assert.equal(trainingIsDue(100, 80), true);
  assert.equal(trainingIsDue(149, 120), false);
  assert.equal(trainingIsDue(150, 120), true);
  assert.equal(trainingIsDue(95, 40), true);
});

test("invalid prior model counts are rejected", () => {
  assert.throws(() => nextTrainingTarget(-1), RangeError);
  assert.throws(() => nextTrainingTarget(1.5), RangeError);
  assert.throws(() => latestTrainingTarget(-1, null), RangeError);
});
