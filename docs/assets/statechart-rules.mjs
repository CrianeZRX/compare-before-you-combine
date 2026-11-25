// The statechart of "One pick from the bin" on the project page, checked by visiting
// every reachable state. Run: node docs/assets/statechart-rules.mjs
// Each rule names the guard that keeps it; removing that guard (adding the arrow it
// keeps shut) must break the rule, and the shortest breaking path is printed.

const MODEL = {
  start: { at: "detected", joint: false, checked: false },
  arrows: [
    { from: "detected", to: "posed", event: "a pose per mask" },
    { from: "posed", to: "scored", event: "one call per mask", set: { joint: false, checked: false } },
    { from: "scored", to: "checked", event: "run the check", set: { checked: true } },
    { from: "checked", to: "scored", event: "score jointly [reversed]", guard: s => !s.joint, set: { joint: true, checked: false } },
    { from: "checked", to: "picked", event: "pool the masks [comparable]", guard: s => s.joint && s.checked },
  ],
  end: "picked",
};
const RULES = [
  { name: "Check before you pool", says: "No mask is picked from scores the check has not read.",
    holds: s => s.at !== "picked" || s.checked,
    without: "checked", extra: { from: "scored", to: "picked", event: "pool the masks, unchecked" } },
  { name: "Never pool per-call scores", says: "No pick compares scores from separate calls.",
    holds: s => s.at !== "picked" || s.joint,
    without: "comparable", extra: { from: "checked", to: "picked", event: "pool the masks, reversed" } },
];

const key = s => JSON.stringify(s);
function next(s, extra) {
  const out = [];
  for (const a of MODEL.arrows.concat(extra ? [extra] : []))
    if (a.from === s.at && (!a.guard || a.guard(s))) out.push([a.event, { ...s, ...(a.set || {}), at: a.to }]);
  return out;
}
function explore(extra) {
  const back = new Map([[key(MODEL.start), null]]), queue = [MODEL.start];
  while (queue.length) {
    const s = queue.shift();
    for (const [ev, t] of next(s, extra)) if (!back.has(key(t))) { back.set(key(t), [s, ev]); queue.push(t); }
  }
  const path = s => { const p = []; for (let c = s; back.get(key(c)); c = back.get(key(c))[0]) p.unshift(back.get(key(c))[1]); return p; };
  return { states: [...back.keys()].map(k => JSON.parse(k)), path };
}

const base = explore();
const places = new Set(MODEL.arrows.flatMap(a => [a.from, a.to]));
const reached = new Set(base.states.map(s => s.at));
for (const p of places) if (!reached.has(p)) throw new Error("never reached: " + p);
for (const s of base.states) {            // from every state a pick can still be made
  const seen = new Set([key(s)]), q = [s]; let ok = s.at === MODEL.end;
  while (q.length && !ok) for (const [, t] of next(q.shift())) { if (t.at === MODEL.end) ok = true; if (!seen.has(key(t))) { seen.add(key(t)); q.push(t); } }
  if (!ok) throw new Error("stuck in " + key(s));
}
const report = [`Every reachable state checked (${base.states.length} states). All ${places.size} steps of the chart are reached, and from each one a pick can still be made.`];
for (const r of RULES) {
  const bad = base.states.find(s => !r.holds(s));
  if (bad) throw new Error(`${r.name} breaks as drawn: ${base.path(bad).join(", ")}`);
  const m = explore(r.extra), broken = m.states.find(s => !r.holds(s));
  if (!broken) throw new Error(`removing [${r.without}] does not break "${r.name}"`);
  const p = m.path(broken);
  report.push(`${r.name}: ${r.says} Without the [${r.without}] guard it breaks in ${p.length} steps: ${p.join(", ")}.`);
}
console.log(report.join("\n"));
