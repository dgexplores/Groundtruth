/* Groundtruth dashboard — Apple-mapped motion.
   Springs: critically damped by default (damping 1.0, response ~0.35s);
   bounce only where a flick precedes motion. All gesture animation starts
   from the live on-screen value and carries release velocity. */

(() => {
  "use strict";

  const REDUCE_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TOLERANCE = 0.05;

  /* ---------- springs (§4) ---------- */

  function animateSpring({ from, to, velocity = 0, damping = 1, response = 0.35, onUpdate, onDone }) {
    if (REDUCE_MOTION) {
      onUpdate(to);
      if (onDone) onDone();
      return () => {};
    }
    const k = (2 * Math.PI / response) ** 2;
    const c = 2 * damping * Math.sqrt(k);
    let x = from;
    let v = velocity;
    let last = performance.now();
    let raf = 0;
    function frame(now) {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const h = dt / 4;
      for (let i = 0; i < 4; i++) {
        v += (-k * (x - to) - c * v) * h;
        x += v * h;
      }
      onUpdate(x);
      if (Math.abs(x - to) < 0.0008 && Math.abs(v) < 0.008) {
        onUpdate(to);
        if (onDone) onDone();
        return;
      }
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }

  /* Soft boundary: progressive resistance, never a hard stop (§9) */
  function rubberband(overshoot, dimension, constant = 0.55) {
    return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
  }

  /* Exponential-decay projection, Apple's form (§6) */
  function project(velocity, rate = 0.998) {
    return velocity * rate / (1 - rate);
  }

  /* ---------- theme (persona: light/dark/system toggle) ---------- */

  const root = document.documentElement;
  const themeButtons = [...document.querySelectorAll(".theme-toggle button")];
  function applyTheme(value) {
    root.dataset.theme = value;
    for (const button of themeButtons) {
      button.setAttribute("aria-pressed", String(button.dataset.themeValue === value));
    }
    try {
      localStorage.setItem("gt-theme", value);
    } catch {
      /* private mode — system default stands */
    }
  }
  for (const button of themeButtons) {
    button.addEventListener("click", () => applyTheme(button.dataset.themeValue));
  }
  let savedTheme = "system";
  try {
    savedTheme = localStorage.getItem("gt-theme") || "system";
  } catch {
    /* ignore */
  }
  applyTheme(savedTheme);

  /* ---------- data ---------- */

  async function loadJSON(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(path);
    return response.json();
  }

  function cellLabel(alias, tier) {
    return `${alias} · ${tier}`;
  }

  function render(report, baseline) {
    const { cells } = report;
    const aliases = Object.keys(cells);

    document.getElementById("corpusMeta").textContent =
      `${report.corpus_size} docs · ${report.total_queries} queries`;

    /* Hero — headline cell */
    const headlineKey = `${report.headline_retriever}_${report.headline_tier}`;
    document.getElementById("headlineLabel").textContent =
      `headline — ${cellLabel(report.headline_retriever, report.headline_tier)}`;
    springNumber(document.getElementById("headlineValue"), report.recall_at_10, 4);
    document.getElementById("headlineR5").textContent =
      `recall@5 ${report.recall_at_5.toFixed(4)}`;

    const pill = document.getElementById("headlinePill");
    const base = baseline ? baseline[`${headlineKey}_recall_at_10`] : null;
    if (base == null) {
      pill.textContent = "no baseline";
    } else {
      const pass = report.recall_at_10 >= base - TOLERANCE;
      pill.textContent = pass ? "gate pass" : "gate fail";
      pill.classList.add(pass ? "pass" : "fail");
    }

    /* Matrix cards */
    const cards = document.getElementById("cards");
    cards.replaceChildren();
    for (const alias of aliases) {
      for (const tier of Object.keys(cells[alias]).sort()) {
        cards.appendChild(matrixCard(alias, tier, cells[alias][tier]));
      }
    }

    /* Compare slider — semantic tier wipe */
    setupCompare(
      report.tfidf_semantic_recall_at_10,
      report.dense_semantic_recall_at_10,
    );

    /* Gates */
    const gates = document.getElementById("gates");
    gates.replaceChildren();
    const lanes = [
      ["headline (dense/semantic)", "recall_at_10"],
      ["tfidf semantic", "tfidf_semantic_recall_at_10"],
    ];
    for (const [name, key] of lanes) {
      gates.appendChild(gateRow(name, baseline ? baseline[key] : null, report[key]));
    }
    if (!baseline) {
      const note = document.createElement("p");
      note.className = "compare-caption";
      note.textContent = "baseline_report.json not found — gates need a baseline.";
      gates.appendChild(note);
    }

    /* Provenance */
    const provenance = document.getElementById("provenance");
    const run = report.run;
    if (run) {
      provenance.innerHTML =
        `run <code>${run.run_id}</code> · <code>${run.git_sha}</code> · ` +
        `${run.started_at} → ${run.finished_at} · ${report.retrievers ? Object.values(report.retrievers).join(" + ") : ""}`;
    } else {
      provenance.textContent =
        "legacy report — no run block. Rerun the benchmark to attach provenance.";
    }
  }

  const live = {}; // current on-screen values, keyed — interrupts start here (§3)

  function springNumber(element, to, digits) {
    const key = element.id || Symbol("n");
    live[key] = live[key] ?? 0;
    const cancel = live[`${String(key)}:cancel`];
    if (cancel) cancel();
    live[`${String(key)}:cancel`] = animateSpring({
      from: live[key],
      to,
      onUpdate: (x) => {
        live[key] = x;
        element.textContent = x.toFixed(digits);
      },
    });
  }

  function springScaleX(element, to) {
    const from = Number(element.dataset.v ?? 0);
    const prev = element._cancel;
    if (prev) prev();
    element._cancel = animateSpring({
      from,
      to,
      onUpdate: (x) => {
        element.dataset.v = x;
        element.style.transform = `scaleX(${Math.max(0, Math.min(1, x))})`;
      },
    });
  }

  function bar(label, value) {
    const wrap = document.createElement("div");
    wrap.className = "bar";
    const head = document.createElement("div");
    head.className = "bar-label";
    head.append(
      Object.assign(document.createElement("span"), { textContent: label }),
      Object.assign(document.createElement("span"), { textContent: value.toFixed(4) }),
    );
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.dataset.v = 0;
    track.appendChild(fill);
    wrap.append(head, track);
    requestAnimationFrame(() => springScaleX(fill, value));
    return wrap;
  }

  function matrixCard(alias, tier, cell) {
    const card = document.createElement("button");
    card.className = "card";
    card.setAttribute("aria-expanded", "false");

    const top = document.createElement("div");
    top.className = "card-top";
    top.append(
      Object.assign(document.createElement("span"), { className: "card-name", textContent: alias }),
      Object.assign(document.createElement("span"), { className: "card-tier", textContent: tier }),
    );
    const value = document.createElement("p");
    value.className = "card-value";
    value.textContent = "–";
    card.append(top, value);
    card.appendChild(bar("recall@10", cell.recall_at_10));
    card.appendChild(bar("recall@5", cell.recall_at_5));

    const detail = document.createElement("div");
    detail.className = "card-detail";
    const inner = document.createElement("div");
    const misses = cell.queries.filter((q) => q.recall_at_10 < 1);
    if (misses.length === 0) {
      inner.appendChild(Object.assign(document.createElement("p"), {
        className: "miss",
        textContent: `all ${cell.total_queries} queries hit at recall@10`,
      }));
    } else {
      for (const miss of misses.slice(0, 8)) {
        const row = document.createElement("p");
        row.className = "miss";
        row.append(
          document.createTextNode(miss.query + " "),
          Object.assign(document.createElement("code"), { textContent: `→ ${miss.id}` }),
        );
        inner.appendChild(row);
      }
      if (misses.length > 8) {
        inner.appendChild(Object.assign(document.createElement("p"), {
          className: "miss",
          textContent: `…and ${misses.length - 8} more`,
        }));
      }
    }
    detail.appendChild(inner);
    card.appendChild(detail);

    card.addEventListener("click", () => {
      card.setAttribute(
        "aria-expanded",
        card.getAttribute("aria-expanded") === "true" ? "false" : "true",
      );
    });

    requestAnimationFrame(() => {
      const from = 0;
      const prev = value._cancel;
      if (prev) prev();
      value._cancel = animateSpring({
        from,
        to: cell.recall_at_10,
        onUpdate: (x) => {
          value.textContent = x.toFixed(4);
        },
      });
    });
    return card;
  }

  function gateRow(name, base, current) {
    const row = document.createElement("div");
    row.className = "gate-row";
    const nameEl = document.createElement("span");
    nameEl.className = "gate-name";
    nameEl.textContent = name;
    row.appendChild(nameEl);
    for (const text of [
      base == null ? "–" : base.toFixed(4),
      current.toFixed(4),
    ]) {
      const cell = document.createElement("span");
      cell.className = "gate-num";
      cell.textContent = text;
      row.appendChild(cell);
    }
    const floor = document.createElement("span");
    floor.className = "gate-num floor";
    floor.textContent = base == null ? "–" : `floor ${(base - TOLERANCE).toFixed(4)}`;
    row.appendChild(floor);
    const pill = document.createElement("span");
    pill.className = "pill";
    if (base == null) {
      pill.textContent = "no baseline";
    } else {
      const pass = current >= base - TOLERANCE;
      pill.textContent = pass ? "pass" : "fail";
      pill.classList.add(pass ? "pass" : "fail");
    }
    row.appendChild(pill);
    return row;
  }

  /* ---------- compare slider: 1:1 wipe with velocity handoff (§2, §5) ---------- */

  function setupCompare(valueA, valueB) {
    const track = document.querySelector("#compare .compare-track");
    const handle = document.getElementById("compareHandle");
    const laneA = track.querySelector(".lane-a");
    document.getElementById("laneAValue").textContent = valueA.toFixed(4);
    document.getElementById("laneBValue").textContent = valueB.toFixed(4);

    let frac = 0.5; // handle position, 0..1 — the live presentation value
    let cancel = null;
    let dragging = false;
    let grabDelta = 0;
    let history = [];

    function paint(f) {
      laneA.style.clipPath = `inset(0 ${(1 - f) * 100}% 0 0)`;
      handle.style.left = `${f * 100}%`;
      handle.setAttribute("aria-valuenow", String(Math.round(f * 100)));
    }
    paint(frac);

    function stop() {
      if (cancel) {
        cancel();
        cancel = null;
      }
    }

    handle.addEventListener("pointerdown", (e) => {
      handle.setPointerCapture(e.pointerId); // track past bounds (§2)
      const rect = track.getBoundingClientRect();
      stop(); // interrupt any glide — grab mid-flight (§3)
      dragging = true;
      grabDelta = (e.clientX - (rect.left + frac * rect.width)) / rect.width;
      history = [{ x: e.clientX, t: performance.now() }];
      e.preventDefault();
    });

    handle.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const rect = track.getBoundingClientRect();
      let f = (e.clientX - rect.left) / rect.width - grabDelta; // respect grab point
      if (f < 0 || f > 1) {
        const bound = f < 0 ? 0 : 1; // rubber-band past the edge (§9)
        f = bound + rubberband(f - bound, 1);
      }
      frac = f;
      paint(frac);
      history.push({ x: e.clientX, t: performance.now() });
      if (history.length > 5) history.shift();
    });

    function releaseVelocity() {
      if (history.length < 2) return 0;
      const first = history[0];
      const last = history[history.length - 1];
      const dt = Math.max((last.t - first.t) / 1000, 0.001);
      const rect = track.getBoundingClientRect();
      return (last.x - first.x) / dt / Math.max(rect.width, 1); // fraction/sec
    }

    handle.addEventListener("pointerup", () => {
      if (!dragging) return;
      dragging = false;
      const v = releaseVelocity();
      // Glide with decay from the live value and velocity — no snap points,
      // the wipe simply coasts and settles where momentum takes it.
      let target = frac + project(v / 1000);
      target = Math.max(0, Math.min(1, target));
      stop();
      cancel = animateSpring({
        from: frac,
        to: target,
        velocity: v,
        damping: 1,
        response: 0.5,
        onUpdate: (x) => {
          frac = Math.max(-0.08, Math.min(1.08, x));
          paint(Math.max(0, Math.min(1, frac)));
        },
      });
    });
    handle.addEventListener("pointercancel", () => {
      dragging = false;
    });

    handle.addEventListener("keydown", (e) => {
      const step = e.key === "ArrowLeft" ? -0.1 : e.key === "ArrowRight" ? 0.1 : 0;
      if (!step) return;
      e.preventDefault();
      stop();
      const to = Math.max(0, Math.min(1, frac + step));
      cancel = animateSpring({ from: frac, to, onUpdate: (x) => { frac = x; paint(frac); } });
    });
  }

  /* ---------- boot ---------- */

  Promise.allSettled([loadJSON("../evals/reports/latest.json"), loadJSON("../evals/baseline_report.json")])
    .then(([latest, baseline]) => {
      if (latest.status !== "fulfilled") {
        document.getElementById("loadError").hidden = false;
        document.querySelector("main").hidden = true;
        return;
      }
      render(
        latest.value,
        baseline.status === "fulfilled" ? baseline.value : null,
      );
    });
})();
