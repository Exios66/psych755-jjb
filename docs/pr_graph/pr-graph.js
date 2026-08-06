/**
 * PR & Issue Progress Graph — vanilla SVG force simulation.
 * Modern physics: symplectic-Euler integration with substeps, grid-hashed
 * many-body charge, timeline spine hinges, collision packing, cluster anchors.
 * Loads baked artifacts/pr_graph/prs.json; no live GitHub calls.
 */
(function () {
  "use strict";

  /* ---------------- Constants ---------------- */

  const KIND = {
    pr: {
      label: "Pull request",
      idPrefix: "PR-",
      shortId: (n) => "PR-" + n.number,
      radius: (n) => 13,
    },
    issue: {
      label: "Issue",
      idPrefix: "I-",
      shortId: (n) => "#" + n.number,
      radius: (n) => 11,
    },
  };

  const STATUS_META = {
    merged: { label: "Merged", color: "#2da44e" },
    open: { label: "Open", color: "#d29922" },
    draft: { label: "Draft", color: "#8b949e" },
    closed: { label: "Closed", color: "#6e7781" },
  };

  // PR statuses shown for issue nodes are open/closed only.
  const PR_STATUSES = ["merged", "open", "draft", "closed"];
  const ISSUE_STATUSES = ["open", "closed"];

  // Resolve a CSS custom property at draw time so the canvas follows the
  // active site scheme (light / dark / osaka-jade / synthwave84).
  function cssVar(name, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (e) {
      return fallback;
    }
  }

  const THEME_PALETTE = [
    "#c5050c",
    "#0f7b8c",
    "#7c3aed",
    "#0891b2",
    "#d97706",
    "#16a34a",
    "#64748b",
    "#db2777",
    "#0f766e",
  ];

  const EDGE_TYPES = {
    timeline: { label: "Timeline" },
    relates: { label: "Relates to" },
    resolves: { label: "Resolves" },
  };

  const DATA_CANDIDATES = [
    "pr_graph/prs.json",
    "./pr_graph/prs.json",
    "../pr_graph/prs.json",
    "../../artifacts/pr_graph/prs.json",
    "../artifacts/pr_graph/prs.json",
    "/artifacts/pr_graph/prs.json",
    "artifacts/pr_graph/prs.json",
  ];

  /* ---------------- Utilities ---------------- */

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function $all(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function parseTime(iso) {
    if (!iso) return null;
    const t = Date.parse(iso);
    return Number.isFinite(t) ? t : null;
  }

  function nodeTime(n) {
    return parseTime(n.mergedAt) || parseTime(n.closedAt) || parseTime(n.createdAt) || 0;
  }

  function formatDate(iso) {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch (_) {
      return String(iso).slice(0, 10);
    }
  }

  function truncate(s, n) {
    s = String(s || "");
    return s.length <= n ? s : s.slice(0, n - 1) + "…";
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function escAttr(s) {
    return esc(s).replace(/'/g, "&#39;");
  }

  function svgEl(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) {
      for (const k in attrs) el.setAttribute(k, String(attrs[k]));
    }
    return el;
  }

  /* ---------------- Force simulation ----------------
   * Symplectic (velocity Verlet-style) Euler integration with substepping:
   *   x += v * dt + 0.5 * a * dt^2 ; a(x_new) ; v += 0.5 * (a_old + a_new) * dt
   * Forces: distance-restrained springs (per edge type), grid-hashed
   * many-body charge, timeline spine hinges, theme cluster anchors, gentle
   * center gravity, and position-based collision packing. */
  function createSimulation(nodes, edges, hinges, opts) {
    const state = {
      nodes,
      edges,
      hinges,
      alpha: 1,
      alphaMin: 0.008,
      alphaDecay: 0.016,
      velocityDecay: 0.84,
      width: opts.width || 800,
      height: opts.height || 600,
      charge: opts.charge || -2600,
      gravity: opts.gravity || 0.06,
      clusterStrength: 0,
      themeAnchors: {},
      linkByType: {
        timeline: { dist: 78, strength: 0.34 },
        relates: { dist: 92, strength: 0.12 },
        resolves: { dist: 64, strength: 0.4 },
      },
      hingeStrength: 0.05,
      collidePad: 10,
      cellSize: 64,
      cutoff: 230,
    };

    function integrate(dt) {
      const { nodes: ns } = state;
      for (let i = 0; i < ns.length; i++) {
        const n = ns[i];
        if (n.fx != null) {
          n.x = n.fx;
          n.vx = 0;
          n.ax = 0;
          n.ay = 0;
          continue;
        }
        // Symplectic Euler: v' = v + a*dt ; x' = x + v'*dt
        n.vx += n.ax * dt;
        n.vy += n.ay * dt;
        n.vx *= state.velocityDecay;
        n.vy *= state.velocityDecay;
        n.x += n.vx * dt;
        n.y += n.vy * dt;
        n.ax = 0;
        n.ay = 0;
      }
    }

    function forces() {
      const { nodes: ns, edges: es, hinges } = state;
      const a = state.alpha;

      // Springs (rest-length restrained, per edge type)
      for (let i = 0; i < es.length; i++) {
        const e = es[i];
        const cfg = state.linkByType[e.type] || state.linkByType.relates;
        const s = e.source;
        const t = e.target;
        let dx = t.x - s.x;
        let dy = t.y - s.y;
        let dist = Math.hypot(dx, dy) || 0.001;
        const f = ((dist - cfg.dist) / dist) * cfg.strength * a;
        dx *= f;
        dy *= f;
        t.vx -= dx;
        t.vy -= dy;
        s.vx += dx;
        s.vy += dy;
      }

      // Timeline spine hinges: straighten consecutive timeline triples.
      if (state.hingeStrength > 0) {
        for (let i = 0; i < hinges.length; i++) {
          const h = hinges[i];
          const { node, a: pa, b: pb } = h;
          const mx = (pa.x + pb.x) / 2;
          const my = (pa.y + pb.y) / 2;
          const k = state.hingeStrength * a;
          node.vx += (mx - node.x) * k;
          node.vy += (my - node.y) * k;
        }
      }

      // Many-body charge via spatial hash (O(n) per frame)
      const cellSize = state.cellSize;
      const grid = new Map();
      for (let i = 0; i < ns.length; i++) {
        const n = ns[i];
        if (n.fx != null) continue;
        const cx = Math.floor(n.x / cellSize);
        const cy = Math.floor(n.y / cellSize);
        const key = cx + ":" + cy;
        let cell = grid.get(key);
        if (!cell) {
          cell = [];
          grid.set(key, cell);
        }
        cell.push(n);
      }
      const cutoff2 = state.cutoff * state.cutoff;
      const chargeK = state.charge * a;
      grid.forEach((cell, key) => {
        for (let i = 0; i < cell.length; i++) {
          for (let j = i + 1; j < cell.length; j++) {
            applyCharge(cell[i], cell[j], chargeK, cutoff2);
          }
        }
        const [cx, cy] = key.split(":").map(Number);
        for (let gx = cx - 1; gx <= cx + 1; gx++) {
          for (let gy = cy - 1; gy <= cy + 1; gy++) {
            if (gx === cx && gy === cy) continue;
            const other = grid.get(gx + ":" + gy);
            if (!other) continue;
            for (let i = 0; i < cell.length; i++) {
              for (let j = 0; j < other.length; j++) {
                applyCharge(cell[i], other[j], chargeK, cutoff2);
              }
            }
          }
        }
      });

      // Theme cluster anchors + center gravity
      for (let i = 0; i < ns.length; i++) {
        const n = ns[i];
        if (n.fx != null) continue;
        const center = state.themeAnchors[n.theme];
        if (state.clusterStrength > 0 && center) {
          n.vx += (center.x - n.x) * state.clusterStrength * a;
          n.vy += (center.y - n.y) * state.clusterStrength * a;
        }
        n.vx += (state.width / 2 - n.x) * state.gravity * a;
        n.vy += (state.height / 2 - n.y) * state.gravity * a;
      }
    }

    function applyCharge(a, b, chargeK, cutoff2) {
      if (a.fx != null || b.fx != null) return;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const dist2 = dx * dx + dy * dy;
      if (dist2 > cutoff2 || dist2 < 0.25) return;
      const dist = Math.sqrt(dist2);
      const f = chargeK / dist2;
      dx = (dx / dist) * f;
      dy = (dy / dist) * f;
      a.vx -= dx;
      a.vy -= dy;
      b.vx += dx;
      b.vy += dy;
    }

    function collide() {
      const { nodes: ns } = state;
      const cellSize = state.cellSize;
      const grid = new Map();
      for (let i = 0; i < ns.length; i++) {
        const n = ns[i];
        const key =
          Math.floor(n.x / cellSize) + ":" + Math.floor(n.y / cellSize);
        let cell = grid.get(key);
        if (!cell) {
          cell = [];
          grid.set(key, cell);
        }
        cell.push(n);
      }
      const pad = state.collidePad;
      grid.forEach((cell, key) => {
        for (let i = 0; i < cell.length; i++) {
          for (let j = i + 1; j < cell.length; j++) {
            resolve(cell[i], cell[j], pad);
          }
        }
        const [cx, cy] = key.split(":").map(Number);
        for (let gx = cx - 1; gx <= cx + 1; gx++) {
          for (let gy = cy - 1; gy <= cy + 1; gy++) {
            if (gx === cx && gy === cy) continue;
            const other = grid.get(gx + ":" + gy);
            if (!other) continue;
            for (let i = 0; i < cell.length; i++) {
              for (let j = 0; j < other.length; j++) {
                resolve(cell[i], other[j], pad);
              }
            }
          }
        }
      });
    }

    function resolve(a, b, pad) {
      const ra = nodeRadius(a);
      const rb = nodeRadius(b);
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const min = ra + rb + pad;
      let dist2 = dx * dx + dy * dy;
      if (dist2 >= min * min || dist2 < 0.001) return;
      const dist = Math.sqrt(dist2);
      const overlap = (min - dist) / dist;
      dx *= overlap * 0.5;
      dy *= overlap * 0.5;
      if (a.fx == null) {
        a.x -= dx;
        a.y -= dy;
      }
      if (b.fx == null) {
        b.x += dx;
        b.y += dy;
      }
    }

    function nodeRadius(n) {
      const kind = KIND[n.kind];
      return kind ? kind.radius(n) : 11;
    }

    function tick() {
      if (state.alpha < state.alphaMin) return false;
      forces();
      // Substep integration for stability at high alpha
      const dt = 0.5;
      integrate(dt);
      integrate(dt);
      collide();
      state.alpha *= 1 - state.alphaDecay;
      return true;
    }

    function reheat(v) {
      state.alpha = Math.max(state.alpha, v == null ? 0.35 : v);
    }

    return { state, tick, reheat };
  }

  function themeAnchorsFor(themes, width, height) {
    const anchors = {};
    const n = Math.max(themes.length, 1);
    const r = Math.min(width, height) * 0.31;
    const cx = width / 2;
    const cy = height / 2;
    themes.forEach((t, i) => {
      const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
      anchors[t] = { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
    });
    return anchors;
  }

  /* ---------------- Data loading ---------------- */

  async function loadDataSmart() {
    const inline = document.getElementById("prg-data");
    if (inline && inline.textContent.trim()) {
      try {
        return JSON.parse(inline.textContent);
      } catch (_) {
        /* fall through to fetch */
      }
    }
    for (const url of DATA_CANDIDATES) {
      try {
        const res = await fetch(url, { cache: "no-cache" });
        if (res.ok) return await res.json();
      } catch (_) {
        /* try next */
      }
    }
    throw new Error("Could not load PR graph data");
  }

  /* ---------------- App ---------------- */

  function buildApp(root, data) {
    const meta = data.meta || {};
    const themeList = (meta.themes || []).map((t) => t.id);
    const themeColors = {};
    themeList.forEach((t, i) => {
      themeColors[t] = THEME_PALETTE[i % THEME_PALETTE.length];
    });

    const nodes = data.nodes.map((n, i) => {
      const angle = (Math.PI * 2 * i) / Math.max(data.nodes.length, 1);
      return {
        ...n,
        x: 400 + Math.cos(angle) * 140,
        y: 300 + Math.sin(angle) * 140,
        vx: 0,
        vy: 0,
        ax: 0,
        ay: 0,
        fx: null,
        fy: null,
        t: nodeTime(n),
      };
    });
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const edges = data.edges
      .filter((e) => byId[e.source] && byId[e.target])
      .map((e) => ({
        ...e,
        source: byId[e.source],
        target: byId[e.target],
      }));

    // Timeline spine hinges: node + its timeline predecessor/successor
    const hinges = [];
    {
      const timeline = edges.filter((e) => e.type === "timeline");
      const pred = new Map();
      const succ = new Map();
      timeline.forEach((e) => {
        succ.set(e.source.id, e.target);
        pred.set(e.target.id, e.source);
      });
      nodes.forEach((n) => {
        const pa = pred.get(n.id);
        const pb = succ.get(n.id);
        if (pa && pb) hinges.push({ node: n, a: pa, b: pb });
      });
    }

    const times = nodes.map((n) => n.t).filter((t) => t > 0);
    const tMin = times.length ? Math.min(...times) : 0;
    const tMax = times.length ? Math.max(...times) : 1;

    const ui = {
      kinds: new Set(["pr", "issue"]),
      edgeType: "all",
      statuses: { pr: new Set(PR_STATUSES), issue: new Set(ISSUE_STATUSES) },
      groupByTheme: false,
      clusterPull: 0.35,
      timeStart: 0,
      timeEnd: 1,
      search: "",
      selectedId: null,
      hoverId: null,
      transform: { x: 0, y: 0, k: 1 },
    };

    const KIND_COUNTS = {
      pr: (meta.prCount != null ? meta.prCount : nodes.filter((n) => n.kind === "pr").length),
      issue: (meta.issueCount != null ? meta.issueCount : nodes.filter((n) => n.kind === "issue").length),
    };

    /* ---------- DOM ---------- */
    root.innerHTML = `
      <div class="prg-app" id="prg-app">
        <div class="prg-backdrop" data-action="close-sidebar" hidden></div>
        <header class="prg-header">
          <div class="prg-brand">psych755-jjb <span>progress</span></div>
          <div class="prg-stats" id="prg-stats"></div>
          <div class="prg-baked" id="prg-baked"></div>
        </header>
        <aside class="prg-sidebar" id="prg-sidebar" aria-label="Items">
          <div class="prg-sidebar-head">
            <h2>Items</h2>
            <span class="prg-count" id="prg-count"></span>
          </div>
          <div class="prg-search">
            <svg class="prg-search-icon" viewBox="0 0 16 16" aria-hidden="true">
              <circle cx="7" cy="7" r="4.6" fill="none" stroke="currentColor" stroke-width="1.6"/>
              <line x1="10.6" y1="10.6" x2="14" y2="14" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
            </svg>
            <input type="search" id="prg-search" placeholder="Search PRs and issues…" autocomplete="off" />
          </div>
          <div class="prg-node-list" id="prg-node-list" role="listbox"></div>
        </aside>
        <section class="prg-main" aria-label="PR & issue relationship graph">
          <div class="prg-mobile-bar">
            <button type="button" data-action="toggle-sidebar">Items</button>
          </div>
          <div class="prg-toolbar">
            <div class="prg-kind-pills" role="group" aria-label="Item kind">
              <button type="button" class="prg-pill is-active" data-kind="pr">Pull requests <span class="prg-pill-count">${KIND_COUNTS.pr}</span></button>
              <button type="button" class="prg-pill is-active" data-kind="issue">Issues <span class="prg-pill-count">${KIND_COUNTS.issue}</span></button>
            </div>
            <div class="prg-pills" role="group" aria-label="Edge type">
              <button type="button" class="prg-pill is-active" data-edge="all">All</button>
              <button type="button" class="prg-pill" data-edge="timeline">Timeline</button>
              <button type="button" class="prg-pill" data-edge="relates">Relates</button>
              <button type="button" class="prg-pill" data-edge="resolves">Resolves</button>
            </div>
            <div class="prg-status-filters" role="group" aria-label="Status filter" id="prg-status-filters"></div>
            <label class="prg-toggle" id="prg-group-toggle">
              <input type="checkbox" id="prg-group" />
              Cluster by theme
            </label>
          </div>
          <div class="prg-canvas-wrap" id="prg-canvas-wrap">
            <svg class="prg-svg" id="prg-svg" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <filter id="prg-soft-shadow" x="-60%" y="-60%" width="220%" height="220%">
                  <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#0b1220" flood-opacity="0.28"/>
                </filter>
                <linearGradient id="prg-grad-merged" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stop-color="${shade(STATUS_META.merged.color, 18)}"/>
                  <stop offset="1" stop-color="${shade(STATUS_META.merged.color, -26)}"/>
                </linearGradient>
                <linearGradient id="prg-grad-open" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stop-color="${shade(STATUS_META.open.color, 18)}"/>
                  <stop offset="1" stop-color="${shade(STATUS_META.open.color, -26)}"/>
                </linearGradient>
                <linearGradient id="prg-grad-closed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stop-color="${shade(STATUS_META.closed.color, 18)}"/>
                  <stop offset="1" stop-color="${shade(STATUS_META.closed.color, -26)}"/>
                </linearGradient>
              </defs>
              <g id="prg-viewport"></g>
            </svg>
          </div>
          <div class="prg-legend" id="prg-legend"></div>
          <div class="prg-zoom" aria-label="Zoom controls">
            <button type="button" data-zoom="in" title="Zoom in">+</button>
            <button type="button" data-zoom="out" title="Zoom out">−</button>
            <button type="button" data-zoom="fit" title="Fit view">⤢</button>
            <button type="button" data-zoom="reset" title="Reset">↺</button>
            <div class="prg-zoom-pct" id="prg-zoom-pct">100%</div>
          </div>
          <div class="prg-sliders">
            <div class="prg-slider">
              <label><span>History</span><span id="prg-time-label">Full history</span></label>
              <input type="range" id="prg-time" min="0" max="1000" value="1000" />
            </div>
            <div class="prg-slider">
              <label><span>Cluster pull</span><span id="prg-pull-label">35%</span></label>
              <input type="range" id="prg-pull" min="0" max="100" value="35" />
            </div>
          </div>
          <div class="prg-tip" id="prg-tip" hidden></div>
          <aside class="prg-drawer" id="prg-drawer" aria-live="polite"></aside>
        </section>
      </div>
    `;

    const app = $("#prg-app", root);
    const svg = $("#prg-svg", root);
    const viewport = $("#prg-viewport", root);
    const canvasWrap = $("#prg-canvas-wrap", root);
    const nodeList = $("#prg-node-list", root);
    const drawer = $("#prg-drawer", root);
    const statsEl = $("#prg-stats", root);
    const bakedEl = $("#prg-baked", root);
    const countEl = $("#prg-count", root);
    const tipEl = $("#prg-tip", root);

    /* Header stats + bake date */
    statsEl.innerHTML =
      '<span class="prg-stat"><i class="prg-dot prg-dot-pr"></i>' +
      KIND_COUNTS.pr +
      " PRs</span>" +
      '<span class="prg-stat"><i class="prg-dot prg-dot-issue"></i>' +
      KIND_COUNTS.issue +
      " issues</span>";
    bakedEl.innerHTML =
      '<span class="prg-bake-chip">Baked ' +
      formatDate(meta.generatedAt) +
      "</span><span class=\"prg-repo\">" +
      esc(meta.repo || "") +
      "</span>";

    /* Status filters (kind-aware) */
    const statusFilters = $("#prg-status-filters", root);
    function buildStatusChips() {
      statusFilters.innerHTML = "";
      const kinds = ["pr", "issue"];
      kinds.forEach((kind) => {
        const group = document.createElement("div");
        group.className = "prg-status-group";
        group.dataset.kind = kind;
        const title = document.createElement("span");
        title.className = "prg-status-group-label";
        title.textContent = kind === "pr" ? "PR" : "Issue";
        group.appendChild(title);
        (kind === "pr" ? PR_STATUSES : ISSUE_STATUSES).forEach((st) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "prg-chip is-active";
          btn.dataset.status = st;
          btn.dataset.kind = kind;
          btn.innerHTML =
            '<span class="prg-status-dot" data-status="' +
            st +
            '" data-kind="' +
            kind +
            '"></span>' +
            STATUS_META[st].label;
          group.appendChild(btn);
        });
        statusFilters.appendChild(group);
      });
    }
    buildStatusChips();

    /* Legend */
    const legend = $("#prg-legend", root);
    function buildLegend() {
      const prStatuses = PR_STATUSES.filter(
        (s) => meta.statusCounts && meta.statusCounts.pr && meta.statusCounts.pr[s]
      );
      const issueStatuses = ISSUE_STATUSES.filter(
        (s) => meta.statusCounts && meta.statusCounts.issue && meta.statusCounts.issue[s]
      );
      legend.innerHTML =
        '<span class="prg-legend-block"><b>PRs</b>' +
        prStatuses
          .map(
            (s) =>
              '<span class="prg-legend-item"><span class="prg-status-dot" data-status="' +
              s +
              '" data-kind="pr"></span>' +
              STATUS_META[s].label +
              "</span>"
          )
          .join("") +
        "</span>" +
        '<span class="prg-legend-block"><b>Issues</b>' +
        issueStatuses
          .map(
            (s) =>
              '<span class="prg-legend-item"><span class="prg-status-dot" data-status="' +
              s +
              '" data-kind="issue"></span>' +
              STATUS_META[s].label +
              "</span>"
          )
          .join("") +
        "</span>";
    }
    buildLegend();

    /* Layers */
    const linkLayer = svgEl("g", { class: "prg-links" });
    const nodeLayer = svgEl("g", { class: "prg-nodes" });
    viewport.appendChild(linkLayer);
    viewport.appendChild(nodeLayer);

    /* Edges as curved paths */
    const linkEls = new Map();
    edges.forEach((e) => {
      const path = svgEl("path", {
        class: "prg-link",
        "data-type": e.type,
        "data-id": e.id,
      });
      linkLayer.appendChild(path);
      linkEls.set(e.id, path);
    });

    /* Nodes */
    const nodeEls = new Map();
    nodes.forEach((n) => {
      const kind = KIND[n.kind];
      const r = kind.radius(n);
      const g = svgEl("g", { class: "prg-node", "data-id": n.id });

      let shape;
      let glyph = null;
      if (n.kind === "pr") {
        const grad =
          n.status === "merged"
            ? "url(#prg-grad-merged)"
            : n.status === "open"
              ? "url(#prg-grad-open)"
              : "url(#prg-grad-closed)";
        shape = svgEl("circle", {
          class: "prg-node-shape",
          r: r,
          cx: 0,
          cy: 0,
          fill: n.status === "draft" ? cssVar("--prg-surface", "#ffffff") : grad,
          stroke: n.status === "draft" ? STATUS_META.draft.color : "none",
          "stroke-width": n.status === "draft" ? 2 : 0,
          "stroke-dasharray": n.status === "draft" ? "2 2" : "",
          filter: "url(#prg-soft-shadow)",
        });
      } else {
        const side = r * 2;
        shape = svgEl("rect", {
          class: "prg-node-shape",
          x: -r,
          y: -r,
          width: side,
          height: side,
          rx: 7,
          ry: 7,
          fill: statusTint(n.status),
          stroke: STATUS_META[n.status].color,
          "stroke-width": 2,
          filter: "url(#prg-soft-shadow)",
        });
        // Issue glyph (appended after the shape below)
        glyph = svgEl("g", { class: "prg-node-glyph" });
        if (n.status === "closed") {
          const check = svgEl("path", {
            d: "M-3 0.5 L-0.8 2.8 L3.2 -2.4",
            fill: "none",
            stroke: STATUS_META.closed.color,
            "stroke-width": 1.8,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          });
          glyph.appendChild(check);
        } else {
          const dot = svgEl("circle", {
            r: 2.2,
            cx: 0,
            cy: 0,
            fill: STATUS_META.open.color,
          });
          glyph.appendChild(dot);
        }
      }

      const themeRing = svgEl("circle", {
        class: "prg-node-theme",
        r: r + 4.5,
        cx: 0,
        cy: 0,
        fill: "none",
        stroke: themeColors[n.theme] || cssVar("--prg-muted", "#94a3b8"),
        "stroke-width": 2.5,
      });

      const idLabel = svgEl("text", {
        class: "prg-node-label prg-label-id",
        y: r + 15,
      });
      idLabel.textContent = kind.shortId(n);

      const titleLabel = svgEl("text", {
        class: "prg-node-label prg-label-title",
        y: r + 15,
      });
      titleLabel.textContent = truncate(n.title, 40);

      g.appendChild(themeRing);
      g.appendChild(shape);
      if (glyph) g.appendChild(glyph);
      g.appendChild(idLabel);
      g.appendChild(titleLabel);
      nodeLayer.appendChild(g);
      nodeEls.set(n.id, g);
    });

    function shade(hex, amt) {
      const h = hex.replace("#", "");
      const num = parseInt(h, 16);
      let r = (num >> 16) & 255;
      let g = (num >> 8) & 255;
      let b = num & 255;
      r = clamp(Math.round(lerp(r, amt > 0 ? 255 : 0, Math.abs(amt) / 100)), 0, 255);
      g = clamp(Math.round(lerp(g, amt > 0 ? 255 : 0, Math.abs(amt) / 100)), 0, 255);
      b = clamp(Math.round(lerp(b, amt > 0 ? 255 : 0, Math.abs(amt) / 100)), 0, 255);
      return "#" + ((r << 16) | (g << 8) | b).toString(16).padStart(6, "0");
    }

    function statusTint(status) {
      const c = STATUS_META[status].color;
      return c + "2b";
    }

    /* Viewport */
    function size() {
      const rect = canvasWrap.getBoundingClientRect();
      const w = Math.max(rect.width, 320);
      const h = Math.max(rect.height, 320);
      svg.setAttribute("viewBox", "0 0 " + w + " " + h);
      svg.setAttribute("width", String(w));
      svg.setAttribute("height", String(h));
      return { width: w, height: h };
    }

    let dims = size();
    nodes.forEach((n, i) => {
      const angle = (Math.PI * 2 * i) / Math.max(nodes.length, 1);
      n.x = dims.width / 2 + Math.cos(angle) * Math.min(dims.width, dims.height) * 0.2;
      n.y = dims.height / 2 + Math.sin(angle) * Math.min(dims.width, dims.height) * 0.2;
    });

    const themes = [...new Set(nodes.map((n) => n.theme))];
    const sim = createSimulation(nodes, edges, hinges, {
      width: dims.width,
      height: dims.height,
    });

    /* ---------- Filters ---------- */
    function kindVisible(n) {
      return ui.kinds.has(n.kind);
    }

    function inTimeRange(n) {
      if (tMax <= tMin) return true;
      const span = tMax - tMin;
      const lo = tMin + ui.timeStart * span;
      const hi = tMin + ui.timeEnd * span;
      return n.t >= lo && n.t <= hi;
    }

    function statusVisible(n) {
      return ui.statuses[n.kind].has(n.status);
    }

    function searchMatch(n) {
      if (!ui.search) return true;
      const q = ui.search.toLowerCase();
      return (
        String(n.number).includes(q) ||
        n.id.toLowerCase().includes(q) ||
        (n.title || "").toLowerCase().includes(q) ||
        (n.themeLabel || "").toLowerCase().includes(q) ||
        (n.author || "").toLowerCase().includes(q) ||
        (n.labels || []).some((l) => String(l).toLowerCase().includes(q))
      );
    }

    function nodeActive(n) {
      return kindVisible(n) && statusVisible(n) && inTimeRange(n) && searchMatch(n);
    }

    function edgeVisible(e) {
      if (ui.edgeType !== "all" && e.type !== ui.edgeType) return false;
      return nodeActive(e.source) && nodeActive(e.target);
    }

    /* ---------- Render ---------- */
    function applyTransform() {
      const { x, y, k } = ui.transform;
      viewport.setAttribute("transform", "translate(" + x + "," + y + ") scale(" + k + ")");
      $("#prg-zoom-pct", root).textContent = Math.round(k * 100) + "%";
      const showTitles = k >= 1.15;
      nodes.forEach((n) => {
        const el = nodeEls.get(n.id);
        if (!el) return;
        el.querySelector(".prg-label-id").style.display = showTitles ? "none" : "";
        el.querySelector(".prg-label-title").style.display = showTitles ? "" : "none";
      });
    }

    function renderList() {
      const visible = nodes
        .filter(nodeActive)
        .sort((a, b) => b.number - a.number);
      countEl.textContent = visible.length + " / " + nodes.length;
      nodeList.innerHTML = "";
      visible.forEach((n) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "prg-node-item";
        if (ui.selectedId === n.id) btn.classList.add("is-selected");
        btn.dataset.id = n.id;
        btn.setAttribute("role", "option");
        btn.innerHTML =
          '<span class="prg-item-icon" data-kind="' +
          n.kind +
          '" data-status="' +
          n.status +
          '"></span>' +
          '<span class="prg-item-body"><span class="prg-node-id">' +
          esc(KIND[n.kind].shortId(n)) +
          ' <em>' +
          esc(n.themeLabel || n.theme) +
          "</em></span>" +
          '<span class="prg-node-title">' +
          esc(truncate(n.title, 80)) +
          "</span></span>";
        btn.addEventListener("click", () => selectNode(n.id, true));
        nodeList.appendChild(btn);
      });
    }

    function paint() {
      const selected = ui.selectedId ? byId[ui.selectedId] : null;
      const hovered = ui.hoverId ? byId[ui.hoverId] : null;
      const focus = selected || hovered;
      const neighborIds = new Set();
      if (focus) {
        edges.forEach((e) => {
          if (!edgeVisible(e)) return;
          if (e.source.id === focus.id) neighborIds.add(e.target.id);
          if (e.target.id === focus.id) neighborIds.add(e.source.id);
        });
        neighborIds.add(focus.id);
      }

      edges.forEach((e) => {
        const el = linkEls.get(e.id);
        if (!el) return;
        const vis = edgeVisible(e);
        el.style.display = vis ? "" : "none";
        const mid = {
          x: (e.source.x + e.target.x) / 2,
          y: (e.source.y + e.target.y) / 2,
        };
        const dx = e.target.x - e.source.x;
        const dy = e.target.y - e.source.y;
        const len = Math.hypot(dx, dy) || 1;
        // Gentle quadratic sag for organic curves
        const sag = e.type === "timeline" ? 0.12 : e.type === "relates" ? 0.16 : 0.08;
        const nx = -dy / len;
        const ny = dx / len;
        const px = mid.x + nx * len * sag * 0.5;
        const py = mid.y + ny * len * sag * 0.5;
        el.setAttribute(
          "d",
          "M" + e.source.x.toFixed(1) + "," + e.source.y.toFixed(1) +
            " Q" + px.toFixed(1) + "," + py.toFixed(1) +
            " " + e.target.x.toFixed(1) + "," + e.target.y.toFixed(1)
        );
        el.classList.toggle(
          "is-dimmed",
          !!(focus && vis && !neighborIds.has(e.source.id))
        );
        el.classList.toggle(
          "is-hot",
          !!(focus && vis && (e.source.id === focus.id || e.target.id === focus.id))
        );
      });

      nodes.forEach((n) => {
        const el = nodeEls.get(n.id);
        if (!el) return;
        const active = nodeActive(n);
        el.style.display = active ? "" : "none";
        el.setAttribute("transform", "translate(" + n.x + "," + n.y + ")");
        el.classList.toggle("is-selected", ui.selectedId === n.id);
        el.classList.toggle(
          "is-dimmed",
          !!(focus && active && !neighborIds.has(n.id))
        );
      });
    }

    /* ---------- Simulation params ---------- */
    function updateSimParams() {
      sim.state.width = dims.width;
      sim.state.height = dims.height;
      sim.state.themeAnchors = themeAnchorsFor(themes, dims.width, dims.height);
      sim.state.clusterStrength = ui.groupByTheme ? ui.clusterPull * 0.09 : 0;
      sim.state.linkByType.timeline.strength = ui.edgeType === "timeline" ? 0.5 : 0.34;
      const activeEdges = edges.filter(edgeVisible);
      sim.state.edges = activeEdges;
      const activeNodes = nodes.filter(nodeActive);
      sim.state.nodes = activeNodes.length ? activeNodes : nodes;
      sim.reheat(0.3);
    }

    let raf = 0;
    function loop() {
      const cont = sim.tick();
      paint();
      if (cont) raf = requestAnimationFrame(loop);
      else raf = 0;
    }

    function kick() {
      updateSimParams();
      if (!raf) raf = requestAnimationFrame(loop);
    }

    /* ---------- Drawer ---------- */
    function openDrawer(n) {
      drawer.classList.add("is-open");
      const kind = KIND[n.kind];
      drawer.innerHTML =
        '<button type="button" class="prg-drawer-close" data-action="close-drawer" aria-label="Close">×</button>' +
        '<div class="prg-drawer-id"><span class="prg-drawer-kind" data-kind="' +
        n.kind +
        '">' +
        (n.kind === "pr" ? "Pull request" : "Issue") +
        '</span> ' +
        esc(kind.shortId(n)) +
        "</div>" +
        "<h3>" +
        esc(n.title) +
        "</h3>" +
        "<div class=\"prg-drawer-tags\">" +
        '<span class="prg-tag" data-status="' +
        n.status +
        '">' +
        STATUS_META[n.status].label +
        "</span>" +
        '<span class="prg-tag prg-tag-theme" style="--prg-tag-color:' +
        (themeColors[n.theme] || cssVar("--prg-muted", "#94a3b8")) +
        '">' +
        esc(n.themeLabel || n.theme) +
        "</span>" +
        (n.labels || [])
          .map((l) => '<span class="prg-tag prg-tag-label">' + esc(l) + "</span>")
          .join("") +
        "</div>" +
        "<dl>" +
        "<dt>Author</dt><dd>" +
        esc(n.author) +
        "</dd>" +
        "<dt>Opened</dt><dd>" +
        formatDate(n.createdAt) +
        "</dd>" +
        (n.kind === "pr"
          ? "<dt>Merged</dt><dd>" + formatDate(n.mergedAt) + "</dd>" +
            "<dt>Closed</dt><dd>" + formatDate(n.closedAt) + "</dd>"
          : "<dt>Closed</dt><dd>" + formatDate(n.closedAt) + "</dd>") +
        "</dl>" +
        (n.excerpt
          ? '<p class="prg-drawer-excerpt">' + esc(n.excerpt) + "</p>"
          : "") +
        '<a class="prg-open-pr" href="' +
        escAttr(n.url) +
        '" target="_blank" rel="noopener noreferrer">Open ' +
        (n.kind === "pr" ? "pull request" : "issue") +
        " ↗</a>";
    }

    function closeDrawer() {
      drawer.classList.remove("is-open");
      drawer.innerHTML = "";
    }

    function selectNode(id, panTo) {
      ui.selectedId = id;
      ui.hoverId = null;
      const n = byId[id];
      if (!n) return;
      openDrawer(n);
      renderList();
      paint();
      if (panTo) {
        const rect = canvasWrap.getBoundingClientRect();
        ui.transform.x = rect.width / 2 - n.x * ui.transform.k;
        ui.transform.y = rect.height / 2 - n.y * ui.transform.k;
        applyTransform();
      }
      const item = nodeList.querySelector('[data-id="' + id + '"]');
      if (item && item.scrollIntoView) {
        item.scrollIntoView({ block: "nearest" });
      }
    }

    /* ---------- Tooltip ---------- */
    function showTip(n, ev) {
      tipEl.innerHTML =
        "<span class=\"prg-tip-id\">" +
        esc(KIND[n.kind].shortId(n)) +
        " · " +
        esc(n.themeLabel || n.theme) +
        "</span>" +
        "<strong>" +
        esc(truncate(n.title, 90)) +
        "</strong>" +
        "<span class=\"prg-tip-meta\">" +
        STATUS_META[n.status].label +
        " · " +
        formatDate(nodeTime(n)) +
        "</span>";
      tipEl.hidden = false;
      positionTip(ev);
    }

    function positionTip(ev) {
      const wrap = canvasWrap.getBoundingClientRect();
      const tx = clamp(ev.clientX - wrap.left + 14, 8, wrap.width - 240);
      const ty = clamp(ev.clientY - wrap.top - 10, 8, wrap.height - 90);
      tipEl.style.left = tx + "px";
      tipEl.style.top = ty + "px";
    }

    function hideTip() {
      tipEl.hidden = true;
    }

    /* ---------- Init ---------- */
    renderList();
    applyTransform();
    kick();

    /* ---------- Interactions ---------- */
    $all(".prg-kind-pills [data-kind]", root).forEach((btn) => {
      btn.addEventListener("click", () => {
        const k = btn.dataset.kind;
        if (ui.kinds.has(k) && ui.kinds.size === 1) return;
        if (ui.kinds.has(k)) ui.kinds.delete(k);
        else ui.kinds.add(k);
        btn.classList.toggle("is-active", ui.kinds.has(k));
        renderList();
        kick();
      });
    });

    $all(".prg-pills [data-edge]", root).forEach((btn) => {
      btn.addEventListener("click", () => {
        $all(".prg-pills [data-edge]", root).forEach((b) =>
          b.classList.remove("is-active")
        );
        btn.classList.add("is-active");
        ui.edgeType = btn.dataset.edge;
        kick();
        paint();
      });
    });

    $all(".prg-chip", root).forEach((btn) => {
      btn.addEventListener("click", () => {
        const st = btn.dataset.status;
        const kind = btn.dataset.kind;
        const set = ui.statuses[kind];
        if (set.has(st) && set.size === 1) return;
        if (set.has(st)) set.delete(st);
        else set.add(st);
        btn.classList.toggle("is-active", set.has(st));
        renderList();
        kick();
      });
    });

    const groupCb = $("#prg-group", root);
    const groupToggle = $("#prg-group-toggle", root);
    groupCb.addEventListener("change", () => {
      ui.groupByTheme = groupCb.checked;
      groupToggle.classList.toggle("is-on", ui.groupByTheme);
      kick();
    });

    $("#prg-search", root).addEventListener("input", (ev) => {
      ui.search = ev.target.value.trim();
      renderList();
      paint();
    });

    const timeSlider = $("#prg-time", root);
    const timeLabel = $("#prg-time-label", root);
    timeSlider.addEventListener("input", () => {
      const v = Number(timeSlider.value) / 1000;
      ui.timeStart = 0;
      ui.timeEnd = clamp(v, 0.02, 1);
      if (ui.timeEnd >= 0.995) {
        timeLabel.textContent = "Full history";
      } else {
        const hi = new Date(tMin + ui.timeEnd * (tMax - tMin));
        timeLabel.textContent = "Through " + formatDate(hi.toISOString());
      }
      renderList();
      kick();
    });

    const pullSlider = $("#prg-pull", root);
    const pullLabel = $("#prg-pull-label", root);
    pullSlider.addEventListener("input", () => {
      ui.clusterPull = Number(pullSlider.value) / 100;
      pullLabel.textContent = Math.round(ui.clusterPull * 100) + "%";
      if (ui.groupByTheme) kick();
    });

    /* Zoom / pan */
    function zoomBy(factor, cx, cy) {
      const rect = canvasWrap.getBoundingClientRect();
      cx = cx == null ? rect.width / 2 : cx;
      cy = cy == null ? rect.height / 2 : cy;
      const prev = ui.transform.k;
      const next = clamp(prev * factor, 0.2, 4);
      ui.transform.x = cx - ((cx - ui.transform.x) * next) / prev;
      ui.transform.y = cy - ((cy - ui.transform.y) * next) / prev;
      ui.transform.k = next;
      applyTransform();
    }

    function fitView() {
      dims = size();
      const active = nodes.filter(nodeActive);
      if (!active.length) return;
      let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;
      active.forEach((n) => {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x);
        maxY = Math.max(maxY, n.y);
      });
      const pad = 56;
      const bw = Math.max(maxX - minX, 40);
      const bh = Math.max(maxY - minY, 40);
      const k = clamp(
        Math.min((dims.width - pad * 2) / bw, (dims.height - pad * 2) / bh),
        0.25,
        2.5
      );
      ui.transform.k = k;
      ui.transform.x = dims.width / 2 - ((minX + maxX) / 2) * k;
      ui.transform.y = dims.height / 2 - ((minY + maxY) / 2) * k;
      applyTransform();
    }

    $all("[data-zoom]", root).forEach((btn) => {
      btn.addEventListener("click", () => {
        const z = btn.dataset.zoom;
        if (z === "in") zoomBy(1.2);
        else if (z === "out") zoomBy(1 / 1.2);
        else if (z === "fit") fitView();
        else if (z === "reset") {
          ui.transform = { x: 0, y: 0, k: 1 };
          applyTransform();
          dims = size();
          nodes.forEach((n, i) => {
            const angle = (Math.PI * 2 * i) / Math.max(nodes.length, 1);
            n.x = dims.width / 2 + Math.cos(angle) * Math.min(dims.width, dims.height) * 0.2;
            n.y = dims.height / 2 + Math.sin(angle) * Math.min(dims.width, dims.height) * 0.2;
            n.vx = 0;
            n.vy = 0;
          });
          kick();
        }
      });
    });

    /* Pointer pan + node drag + hover */
    let pan = null;
    let drag = null;
    let lastTap = 0;

    canvasWrap.addEventListener("pointermove", (ev) => {
      if (drag || pan) return;
      const nodeEl = ev.target.closest ? ev.target.closest(".prg-node") : null;
      if (nodeEl) {
        const id = nodeEl.dataset.id;
        if (ui.hoverId !== id) {
          ui.hoverId = id;
          const n = byId[id];
          if (n) showTip(n, ev);
          paint();
        } else {
          positionTip(ev);
        }
      } else if (ui.hoverId) {
        ui.hoverId = null;
        hideTip();
        paint();
      }
    });

    canvasWrap.addEventListener("pointerdown", (ev) => {
      const nodeEl = ev.target.closest ? ev.target.closest(".prg-node") : null;
      if (nodeEl) {
        const g = nodeEl;
        const id = g.dataset.id;
        const n = byId[id];
        const now = Date.now();
        if (now - lastTap < 350 && ui.selectedId === id) {
          window.open(n.url, "_blank", "noopener,noreferrer");
          lastTap = 0;
          return;
        }
        lastTap = now;
        selectNode(id, false);
        drag = {
          id,
          pointerId: ev.pointerId,
          ox: ev.clientX,
          oy: ev.clientY,
        };
        n.fx = n.x;
        n.fy = n.y;
        canvasWrap.setPointerCapture(ev.pointerId);
        sim.reheat(0.3);
        if (!raf) raf = requestAnimationFrame(loop);
        ev.preventDefault();
        return;
      }
      pan = {
        pointerId: ev.pointerId,
        x: ev.clientX,
        y: ev.clientY,
        tx: ui.transform.x,
        ty: ui.transform.y,
      };
      canvasWrap.classList.add("is-panning");
      canvasWrap.setPointerCapture(ev.pointerId);
    });

    canvasWrap.addEventListener("pointermove", (ev) => {
      if (drag && ev.pointerId === drag.pointerId) {
        const n = byId[drag.id];
        const dx = (ev.clientX - drag.ox) / ui.transform.k;
        const dy = (ev.clientY - drag.oy) / ui.transform.k;
        n.fx = n.x + dx;
        n.fy = n.y + dy;
        n.x = n.fx;
        n.y = n.fy;
        drag.ox = ev.clientX;
        drag.oy = ev.clientY;
        paint();
        sim.reheat(0.15);
        if (!raf) raf = requestAnimationFrame(loop);
        return;
      }
      if (pan && ev.pointerId === pan.pointerId) {
        ui.transform.x = pan.tx + (ev.clientX - pan.x);
        ui.transform.y = pan.ty + (ev.clientY - pan.y);
        applyTransform();
      }
    });

    function endPointer(ev) {
      if (drag && ev.pointerId === drag.pointerId) {
        const n = byId[drag.id];
        n.fx = null;
        n.fy = null;
        drag = null;
        sim.reheat(0.25);
        if (!raf) raf = requestAnimationFrame(loop);
      }
      if (pan && ev.pointerId === pan.pointerId) {
        pan = null;
        canvasWrap.classList.remove("is-panning");
      }
    }

    canvasWrap.addEventListener("pointerup", endPointer);
    canvasWrap.addEventListener("pointercancel", endPointer);

    canvasWrap.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const rect = canvasWrap.getBoundingClientRect();
        const factor = ev.deltaY < 0 ? 1.08 : 1 / 1.08;
        zoomBy(factor, ev.clientX - rect.left, ev.clientY - rect.top);
      },
      { passive: false }
    );

    /* Pinch zoom */
    let pinch = null;
    canvasWrap.addEventListener(
      "touchstart",
      (ev) => {
        if (ev.touches.length === 2) {
          const [a, b] = ev.touches;
          pinch = {
            dist: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY),
            k: ui.transform.k,
          };
        }
      },
      { passive: true }
    );
    canvasWrap.addEventListener(
      "touchmove",
      (ev) => {
        if (ev.touches.length === 2 && pinch) {
          const [a, b] = ev.touches;
          const dist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
          const next = clamp(pinch.k * (dist / pinch.dist), 0.2, 4);
          const rect = canvasWrap.getBoundingClientRect();
          const cx = (a.clientX + b.clientX) / 2 - rect.left;
          const cy = (a.clientY + b.clientY) / 2 - rect.top;
          const prev = ui.transform.k;
          ui.transform.x = cx - ((cx - ui.transform.x) * next) / prev;
          ui.transform.y = cy - ((cy - ui.transform.y) * next) / prev;
          ui.transform.k = next;
          applyTransform();
        }
      },
      { passive: true }
    );
    canvasWrap.addEventListener(
      "touchend",
      () => {
        pinch = null;
      },
      { passive: true }
    );

    root.addEventListener("click", (ev) => {
      const t = ev.target;
      if (!(t instanceof Element)) return;
      if (t.closest("[data-action='close-drawer']")) {
        ui.selectedId = null;
        closeDrawer();
        paint();
        renderList();
      }
      if (t.closest("[data-action='toggle-sidebar']")) {
        app.classList.toggle("is-sidebar-open");
        const bd = $(".prg-backdrop", root);
        if (bd) bd.hidden = !app.classList.contains("is-sidebar-open");
      }
      if (t.closest("[data-action='close-sidebar']")) {
        app.classList.remove("is-sidebar-open");
        const bd = $(".prg-backdrop", root);
        if (bd) bd.hidden = true;
      }
    });

    const ro = new ResizeObserver(() => {
      dims = size();
      sim.state.width = dims.width;
      sim.state.height = dims.height;
      sim.state.themeAnchors = themeAnchorsFor(themes, dims.width, dims.height);
      sim.reheat(0.2);
      if (!raf) raf = requestAnimationFrame(loop);
    });
    ro.observe(canvasWrap);

    document.body.classList.add("pr-graph-page");
    document.documentElement.classList.add("pr-graph-page");
  }

  async function boot() {
    const root = document.getElementById("pr-progress-graph");
    if (!root) return;
    root.innerHTML = '<div class="prg-loading">Loading progress graph…</div>';
    try {
      const data = await loadDataSmart();
      if (!data || !Array.isArray(data.nodes)) {
        throw new Error("Invalid graph payload");
      }
      buildApp(root, data);
    } catch (err) {
      console.error(err);
      root.innerHTML =
        '<div class="prg-error">Could not load graph data. ' +
        "Ensure <code>artifacts/pr_graph/prs.json</code> is published with the site. " +
        "<br/><small>" +
        String(err && err.message ? err.message : err) +
        "</small></div>";
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
