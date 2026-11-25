/* Cross-call pairwise accuracy in the browser: a line-by-line port of
   src/compare6d/check.py (v1.1, six-image rule). Python's random module is
   reproduced (MT19937 with the same seeding), so for the same table, draws and
   seed the result equals `compare6d check --table` number for number.
   Nothing is uploaded: the table is read and checked on this page. */
(function (root) {
  'use strict';
  var ALPHA = 0.05, MIN_IMAGES = 6, EXACT_UP_TO = 16, FLIP_DRAWS = 20000, TOLERANCE = 1e-12;
  var TRUE = { '1': 1, 'true': 1, 't': 1, 'yes': 1 }, FALSE = { '0': 1, 'false': 1, 'f': 1, 'no': 1 };

  /* ---- Python's random.Random(seed): MT19937, init_by_array, random(), choices(), getrandbits() ---- */
  function PyRandom(seed) {
    var N = 624, mt = new Uint32Array(N), mti = N + 1, i, j, k;
    function initGenrand(s) { mt[0] = s >>> 0; for (mti = 1; mti < N; mti++) { var p = mt[mti - 1] ^ (mt[mti - 1] >>> 30); mt[mti] = (Math.imul(1812433253, p) + mti) >>> 0; } }
    var key = [];
    var n = Math.abs(Math.floor(seed || 0));
    if (n === 0) key = [0]; else while (n > 0) { key.push(n % 4294967296); n = Math.floor(n / 4294967296); }
    initGenrand(19650218);
    i = 1; j = 0; k = N > key.length ? N : key.length;
    for (; k; k--) { var q = mt[i - 1] ^ (mt[i - 1] >>> 30); mt[i] = ((mt[i] ^ Math.imul(q, 1664525)) + key[j] + j) >>> 0; i++; j++; if (i >= N) { mt[0] = mt[N - 1]; i = 1; } if (j >= key.length) j = 0; }
    for (k = N - 1; k; k--) { var r = mt[i - 1] ^ (mt[i - 1] >>> 30); mt[i] = ((mt[i] ^ Math.imul(r, 1566083941)) - i) >>> 0; i++; if (i >= N) { mt[0] = mt[N - 1]; i = 1; } }
    mt[0] = 0x80000000;
    function u32() {
      var y, kk;
      if (mti >= N) {
        for (kk = 0; kk < N - 397; kk++) { y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff); mt[kk] = mt[kk + 397] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0); }
        for (; kk < N - 1; kk++) { y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff); mt[kk] = mt[kk + (397 - N)] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0); }
        y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff); mt[N - 1] = mt[396] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
        mti = 0;
      }
      y = mt[mti++];
      y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
      return y >>> 0;
    }
    return {
      random: function () { var a = u32() >>> 5, b = u32() >>> 6; return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0); },
      /* the bits of getrandbits(k) as 32-bit words, least significant first */
      bitWords: function (k) {
        if (k <= 32) return [k ? u32() >>> (32 - k) : 0];
        var words = Math.floor((k - 1) / 32) + 1, out = [];
        for (var w = 0; w < words; w++, k -= 32) { var v = u32(); if (k < 32) v = v >>> (32 - k); out.push(v); }
        return out;
      }
    };
  }

  function bisectLeft(a, x) { var lo = 0, hi = a.length; while (lo < hi) { var m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; }
  function bisectRight(a, x) { var lo = 0, hi = a.length; while (lo < hi) { var m = (lo + hi) >> 1; if (x < a[m]) hi = m; else lo = m + 1; } return lo; }
  function subsetSums(values) { var sums = [0.0]; for (var i = 0; i < values.length; i++) { var v = values[i], a = sums.map(function (x) { return x - v; }), b = sums.map(function (x) { return x + v; }); sums = a.concat(b); } return sums; }

  function signFlipP(dev, seed) {
    var n = dev.length, s = 0, i;
    for (i = 0; i < n; i++) s += dev[i];
    var observed = Math.abs(s);
    if (observed <= TOLERANCE) return 1.0;
    if (n <= EXACT_UP_TO) {
      var half = n >> 1, low = subsetSums(dev.slice(0, half)), ordered = subsetSums(dev.slice(half)).sort(function (a, b) { return a - b; }), count = 0;
      for (i = 0; i < low.length; i++) { var a = low[i]; count += ordered.length - bisectLeft(ordered, observed - a - TOLERANCE); count += bisectRight(ordered, -observed - a + TOLERANCE); }
      return count / Math.pow(2, n);
    }
    var tables = []; for (i = 0; i < n; i += 8) tables.push(subsetSums(dev.slice(i, i + 8)));
    var rng = PyRandom(seed), hits = 0;
    for (var d = 0; d < FLIP_DRAWS; d++) {
      var words = rng.bitWords(n), total = 0.0;
      for (var t = 0; t < tables.length; t++) total += tables[t][(words[t >> 2] >>> (8 * (t & 3))) & 255];
      if (Math.abs(total) >= observed - TOLERANCE) hits++;
    }
    return (1 + hits) / (1 + FLIP_DRAWS);
  }

  /* rows: [{image: string key, obj: string, score: number, correct: boolean}] in file order */
  function check(rows, draws, seed) {
    draws = draws == null ? 2000 : draws; seed = seed == null ? 0 : seed;
    var targets = new Map(), perImage = new Map(), i;
    rows.forEach(function (r) {
      if (!isFinite(r.score)) throw new Error('Scores must be finite.');
      var key = JSON.stringify([r.image, r.obj]);
      if (!targets.has(key)) targets.set(key, { image: r.image, scores: [], labels: [] });
      var t = targets.get(key); t.scores.push(r.score); t.labels.push(!!r.correct);
    });
    targets.forEach(function (t) {
      var w = 0, ti = 0, p = 0;
      for (var a = 0; a < t.scores.length; a++) for (var b = a + 1; b < t.scores.length; b++) {
        if (t.labels[a] === t.labels[b]) continue;
        var pos = t.labels[a] ? t.scores[a] : t.scores[b], neg = t.labels[a] ? t.scores[b] : t.scores[a];
        p++; if (pos > neg) w++; if (pos === neg) ti++;
      }
      var cur = perImage.get(t.image) || { wins: 0, ties: 0, pairs: 0 };
      perImage.set(t.image, { wins: cur.wins + w, ties: cur.ties + ti, pairs: cur.pairs + p });
    });
    var total = { wins: 0, ties: 0, pairs: 0 }, counted = [];
    perImage.forEach(function (v) { total.wins += v.wins; total.ties += v.ties; total.pairs += v.pairs; if (v.pairs) counted.push(v); });
    if (!total.pairs) throw new Error('No image and object has both a correct and an incorrect pose, so there is nothing to compare.');
    var rng = PyRandom(seed), samples = new Float64Array(draws), m = counted.length;
    for (var d = 0; d < draws; d++) {
      var credit = 0.0, pairs = 0.0;
      for (i = 0; i < m; i++) { var v = counted[Math.floor(rng.random() * m)]; credit += v.wins + 0.5 * v.ties; pairs += v.pairs; }
      samples[d] = credit / pairs;
    }
    samples.sort();
    var low = samples[Math.floor(0.025 * draws)], high = samples[Math.min(draws - 1, Math.floor(0.975 * draws))];
    var dev = counted.map(function (v) { return v.wins + 0.5 * v.ties - 0.5 * v.pairs; });
    var p = signFlipP(dev, seed), reading, advice;
    if (total.ties === total.pairs) { reading = 'no_information'; advice = 'Every correct pose has the same score as the incorrect pose it is paired with, so these scores say nothing about which pose is correct. Use another score.'; }
    else if (m < MIN_IMAGES) { reading = 'too_few_images'; advice = 'Only ' + m + ' image(s) hold both a correct and an incorrect pose of one object; at least ' + MIN_IMAGES + ' are needed before any direction can be told from chance. Label more images.'; }
    else if (p >= ALPHA) { reading = 'not_comparable'; advice = 'The images do not agree on a direction. These scores cannot be compared across calls on this evidence, and normalizing each call is no substitute. Use more labelled images or another score.'; }
    else if (high < 0.5) { reading = 'reversed'; advice = 'Across calls these scores order pairs worse than chance. Do not compare or submit them as they are. Score the candidates of a target in one call, use a score that rates each pose on its own, or use the reversed score after checking its sign on each new detector and dataset.'; }
    else if (low > 0.5) { reading = 'comparable'; advice = 'Across calls these scores point the right way on these data. The accuracy says how reliably.'; }
    else { reading = 'not_comparable'; advice = 'The interval includes 50%. These scores cannot be compared across calls on this evidence, and normalizing each call is no substitute. Use more labelled images or another score.'; }
    return {
      pairs: total.pairs, wins: total.wins, ties: total.ties, tie_rate: total.ties / total.pairs,
      accuracy: (total.wins + 0.5 * total.ties) / total.pairs, accuracy_strict: total.wins / total.pairs,
      interval_95: [low, high], images_with_pairs: m, images: perImage.size,
      sign_flip_p: p, min_images: MIN_IMAGES,
      bootstrap: { unit: 'image', draws: draws, seed: seed, ties: 'half credit' },
      reading: reading, advice: advice
    };
  }

  /* CSV with the columns scene_id, im_id, obj_id, score, correct (and an optional dataset column) */
  function parseCSV(text) {
    var recs = [], row = [], f = '', q = false, c, i;
    text = String(text).replace(/^﻿/, '');
    for (i = 0; i < text.length; i++) {
      c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; continue; }
      if (c === '"') q = true; else if (c === ',') { row.push(f); f = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); f = ''; recs.push(row); row = []; }
      else f += c;
    }
    if (f !== '' || row.length) { row.push(f); recs.push(row); }
    recs = recs.filter(function (r) { return !(r.length === 1 && r[0] === ''); });
    if (!recs.length) throw new Error('The table is empty.');
    var head = recs[0], need = ['correct', 'im_id', 'obj_id', 'scene_id', 'score'], col = {};
    head.forEach(function (h, k) { if (!(h in col)) col[h] = k; });
    if (need.some(function (h) { return !(h in col); })) throw new Error('The table needs the columns ' + need.join(', ') + '.');
    return recs.slice(1).map(function (r, n) {
      var get = function (h) { var v = r[col[h]]; if (v === undefined) throw new Error('Row ' + (n + 2) + ' has too few fields.'); return v; };
      var label = get('correct').trim().toLowerCase();
      if (!(label in TRUE) && !(label in FALSE)) throw new Error('Unreadable correctness label: ' + JSON.stringify(get('correct')));
      var raw = get('score').trim(), score = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(raw) ? parseFloat(raw) : (/^[+-]?(inf|infinity|nan)$/i.test(raw) ? NaN : undefined);
      if (score === undefined) throw new Error('Unreadable score on row ' + (n + 2) + ': ' + JSON.stringify(raw));
      if (!isFinite(score)) throw new Error('Scores must be finite.');
      var ds = 'dataset' in col ? (r[col.dataset] === undefined ? '' : r[col.dataset]) : '';
      return { image: JSON.stringify([ds, get('scene_id').trim(), get('im_id').trim()]), obj: get('obj_id').trim(), score: score, correct: label in TRUE };
    });
  }

  var api = { check: check, parseCSV: parseCSV, signFlipP: signFlipP, PyRandom: PyRandom, MIN_IMAGES: MIN_IMAGES };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.compare6dCheck = api;
})(this);
