import assert from "node:assert/strict";
import test from "node:test";
import { curatorAllowance, escapeLabel, pairwiseReferencePhotos } from "../lib/curator-policy";

test("curation waits until at most fifty uncompared remain", () => {
  assert.equal(curatorAllowance(51,0),0);
  assert.equal(curatorAllowance(50,0),10);
  assert.equal(curatorAllowance(0,0),10);
});
test("bootstrap permits first batch but never bypasses daily cap", () => {
  assert.equal(curatorAllowance(60,0,true),10);
  assert.equal(curatorAllowance(60,97,true),3);
  assert.equal(curatorAllowance(5,100),0);
  assert.throws(() => curatorAllowance(-1,0));
  assert.throws(() => curatorAllowance(1,NaN));
});
test("contact-sheet labels cannot inject SVG", () => {
  assert.equal(escapeLabel('<script x="&">'), '&lt;script x=&quot;&amp;&quot;&gt;');
});
test("pairwise references include recent opponents and exclude Elo-only seeds", () => {
  const photos = [
    { id: 1, elo: 2000, matches: 0 },
    { id: 2, elo: 1600, matches: 2 },
    { id: 3, elo: 1400, matches: 3 },
    { id: 4, elo: 1450, matches: 1 },
  ];
  const selected = pairwiseReferencePhotos(photos, [{ left_id: 4, right_id: 2, winner_id: 4 }]);
  assert.deepEqual(selected.map((photo) => photo.id), [4, 2, 3]);
  assert.equal(new Set(selected.map((photo) => photo.id)).size, selected.length);
  assert.deepEqual(pairwiseReferencePhotos([photos[0]], []), []);
});
