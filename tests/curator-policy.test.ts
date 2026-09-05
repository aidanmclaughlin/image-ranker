import assert from "node:assert/strict";
import test from "node:test";
import { curatorAllowance, escapeLabel } from "../lib/curator-policy";

test("curation waits until at most fifty unrated remain", () => {
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
