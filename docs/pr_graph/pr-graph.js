/**
 * PR Progress Graph — vanilla SVG force simulation.
 * Loads baked artifacts/pr_graph/prs.json; no live GitHub calls.
 */
(function () {
  "use strict";

  const STATUS_COLORS = {
    merged: "#1a7f37",
    open: "#bf6a00",
    draft: "#6e7781",
    closed: "#8b949e",
  };

  const STATUS_LABELS = {
    merged: "Merged",
    open: "Open",
    draft: "Draft",
    closed: "Closed",
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

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function $all(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
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
      return iso.slice(0, 10);
    }
  }

  function truncate(s, n) {
    s = String(s || "");
    return s.length <= n ? s : s.slice(0, n - 1) + "…";
  }

  /* —— Minimal force simulation —— */
  function createSimulation(nodes, edges, opts) {
    const state = {
      nodes,
      edges,
      alpha: 1,
      alphaMin: 0.001,
      alphaDecay: 0.028,
      velocityDecay: 0.65,
      linkDistance: opts.linkDistance || 70,
      linkStrength: opts.linkStrength || 0.35,
      charge: opts.charge || -180,
      centerStrength: opts.centerStrength || 0.05,
      clusterStrength: opts.clusterStrength || 0,
      width: opts.width || 800,
      height: opts.height || 600,
      themeCenters: opts.themeCenters || {},
    };

    function tick() {
      if (state.alpha < state.alphaMin) return false;

      const { nodes: ns, edges: es } = state;
      const cx = state.width / 2;
      const cy = state.height / 2;

      // Links
      for (let i = 0; i < es.length; i++) {
        const e = es[i];
        const s = e.source;
        const t = e.target;
        let dx = t.x - s.x;
        let dy = t.y - s.y;
        let dist = Math.hypot(dx, dy) || 1;
        const force = ((dist - state.linkDistance) / dist) * state.linkStrength * state.alpha;
        dx *= force;
        dy *= force;
        t.vx -= dx;
        t.vy -= dy;
        s.vx += dx;
        s.vy += dy;
      }

      // Charge (n^2 fine for ~50 nodes)
      for (let i = 0; i < ns.length; i++) {
        for (let j = i + 1; j < ns.length; j++) {
          const a = ns[i];
          const b = ns[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let dist2 = dx * dx + dy * dy || 1;
          if (dist2 > 250000) continue;
          const dist = Math.sqrt(dist2);
          const force = (state.charge * state.alpha) / dist2;
          dx = (dx / dist) * force;
          dy = (dy / dist) * force;
          a.vx -= dx;
          a.vy -= dy;
          b.vx += dx;
          b.vy += dy;
        }
      }

      // Center + optional theme clustering
      for (let i = 0; i < ns.length; i++) {
        const n = ns[i];
        n.vx += (cx - n.x) * state.centerStrength * state.alpha;
        n.vy += (cy - n.y) * state.centerStrength * state.alpha;
        if (state.clusterStrength > 0 && state.themeCenters[n.theme]) {
          const tc = state.themeCenters[n.theme];
          n.vx += (tc.x - n.x) * state.clusterStrength * state.alpha;
          n.vy += (tc.y - n.y) * state.clusterStrength * state.alpha;
        }
      }

      for (let i = 0; i < ns.length; i++) {
        const n = ns[i];
        if (n.fx != null) {
          n.x = n.fx;
          n.vx = 0;
        } else {
          n.vx *= state.velocityDecay;
          n.x += n.vx;
        }
        if (n.fy != null) {
          n.y = n.fy;
          n.vy = 0;
        } else {
          n.vy *= state.velocityDecay;
          n.y += n.vy;
        }
      }

      state.alpha *= 1 - state.alphaDecay;
      return true;
    }

    function reheat(a) {
      state.alpha = Math.max(state.alpha, a == null ? 0.4 : a);
    }

    return { state, tick, reheat };
  }

  function themeCentersFor(themes, width, height) {
    const centers = {};
    const n = Math.max(themes.length, 1);
    const r = Math.min(width, height) * 0.28;
    const cx = width / 2;
    const cy = height / 2;
    themes.forEach((t, i) => {
      const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
      centers[t] = { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
    });
    return centers;
  }

  async function loadData() {
    let lastErr = null;
    for (const url of DATA_CANDIDATES) {
      try {
        const res = await fetch(url, { cache: "no-cache" });
        if (!res.ok) {
          lastErr = new Error(url + " → " + res.status);
          continue;
        }
        return await res.json();
      } catch (err) {
        lastErr = err;
      }
    }
    // Inline fallback if Quarto embedded data
    const inline = document.getElementById("prg-data");
    if (inline && inline.textContent.trim()) {
      return JSON.parse(inline.textContent);
    }
    throw lastErr || new Error("Could not load PR graph data");
  }

  function resolveDataUrl() {
    const el = document.getElementById("pr-progress-graph");
    return el && el.dataset && el.dataset.prsUrl ? el.dataset.prsUrl : null;
  }

  async function loadDataSmart() {
    // Prefer inline embed (always available after Quarto render on Posit).
    const inline = document.getElementById("prg-data");
    if (inline && inline.textContent.trim()) {
      try {
        return JSON.parse(inline.textContent);
      } catch (_) {
        /* fall through to fetch */
      }
    }
    const preferred = resolveDataUrl();
    if (preferred) {
      try {
        const res = await fetch(preferred, { cache: "no-cache" });
        if (res.ok) return await res.json();
      } catch (_) {
        /* fall through */
      }
    }
    return loadData();
  }

  function buildApp(root, data) {
    const nodes = data.nodes.map((n, i) => {
      const angle = (Math.PI * 2 * i) / data.nodes.length;
      return {
        ...n,
        x: 400 + Math.cos(angle) * 120,
        y: 300 + Math.sin(angle) * 120,
        vx: 0,
        vy: 0,
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

    const times = nodes.map((n) => n.t).filter((t) => t > 0);
    const tMin = times.length ? Math.min(...times) : 0;
    const tMax = times.length ? Math.max(...times) : 1;

    const ui = {
      edgeType: "all",
      statuses: new Set(["merged", "open", "draft", "closed"]),
      groupByTheme: false,
      clusterPull: 0.35,
      timeStart: 0,
      timeEnd: 1,
      search: "",
      selectedId: null,
      transform: { x: 0, y: 0, k: 1 },
    };

    root.innerHTML = `
      <div class="prg-app" id="prg-app">
        <div class="prg-backdrop" data-action="close-sidebar" hidden></div>
        <header class="prg-header">
          <div class="prg-brand">psych755-jjb <span>PR progress</span></div>
          <div class="prg-tabs" aria-hidden="true">
            <span class="prg-tab is-active">Graph</span>
          </div>
          <div class="prg-meta" id="prg-meta"></div>
        </header>
        <aside class="prg-sidebar" id="prg-sidebar" aria-label="PR nodes">
          <div class="prg-sidebar-head">
            <h2>Nodes</h2>
            <span class="prg-count" id="prg-count"></span>
          </div>
          <div class="prg-search">
            <input type="search" id="prg-search" placeholder="Search PRs…" autocomplete="off" />
          </div>
          <div class="prg-node-list" id="prg-node-list" role="listbox"></div>
        </aside>
        <section class="prg-main" aria-label="PR relationship graph">
          <div class="prg-mobile-bar">
            <button type="button" data-action="toggle-sidebar">Nodes</button>
          </div>
          <div class="prg-toolbar">
            <div class="prg-pills" role="group" aria-label="Edge type">
              <button type="button" class="prg-pill is-active" data-edge="all">All edges</button>
              <button type="button" class="prg-pill" data-edge="timeline">Timeline</button>
              <button type="button" class="prg-pill" data-edge="relates">Relates to</button>
            </div>
            <div class="prg-status-filters" role="group" aria-label="Status filter" id="prg-status-filters"></div>
            <label class="prg-toggle" id="prg-group-toggle">
              <input type="checkbox" id="prg-group" />
              Group by theme
            </label>
          </div>
          <div class="prg-canvas-wrap" id="prg-canvas-wrap">
            <svg class="prg-svg" id="prg-svg" xmlns="http://www.w3.org/2000/svg">
              <g id="prg-viewport"></g>
            </svg>
          </div>
          <div class="prg-legend" id="prg-legend"></div>
          <div class="prg-zoom" aria-label="Zoom controls">
            <button type="button" data-zoom="in" title="Zoom in">+</button>
            <button type="button" data-zoom="out" title="Zoom out">−</button>
            <button type="button" data-zoom="fit" title="Fit">⊙</button>
            <button type="button" data-zoom="reset" title="Reset">↺</button>
            <div class="prg-zoom-pct" id="prg-zoom-pct">100%</div>
          </div>
          <div class="prg-sliders">
            <div class="prg-slider">
              <label><span>Time range</span><span id="prg-time-label">Full history</span></label>
              <input type="range" id="prg-time" min="0" max="1000" value="1000" />
            </div>
            <div class="prg-slider">
              <label><span>Cluster pull</span><span id="prg-pull-label">35%</span></label>
              <input type="range" id="prg-pull" min="0" max="100" value="35" />
            </div>
          </div>
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
    const metaEl = $("#prg-meta", root);
    const countEl = $("#prg-count", root);

    metaEl.textContent =
      (data.meta.nodeCount || nodes.length) +
      " PRs · " +
      (data.meta.edgeCount || edges.length) +
      " links · baked " +
      formatDate(data.meta.generatedAt) +
      " · " +
      (data.meta.repo || "");

    // Status filters
    const statusFilters = $("#prg-status-filters", root);
    ["merged", "open", "draft", "closed"].forEach((st) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "prg-chip is-active";
      btn.dataset.status = st;
      btn.innerHTML =
        '<span class="prg-status-dot" data-status="' +
        st +
        '"></span>' +
        STATUS_LABELS[st];
      statusFilters.appendChild(btn);
    });

    // Legend
    const legend = $("#prg-legend", root);
    legend.innerHTML = ["merged", "open", "draft", "closed"]
      .map(
        (st) =>
          '<span class="prg-legend-item"><span class="prg-status-dot" data-status="' +
          st +
          '"></span>' +
          STATUS_LABELS[st].toUpperCase() +
          "</span>"
      )
      .join("");

    const linkLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    linkLayer.setAttribute("class", "prg-links");
    const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodeLayer.setAttribute("class", "prg-nodes");
    viewport.appendChild(linkLayer);
    viewport.appendChild(nodeLayer);

    const linkEls = new Map();
    edges.forEach((e) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "line");
      path.setAttribute("class", "prg-link");
      path.dataset.type = e.type;
      path.dataset.id = e.id;
      linkLayer.appendChild(path);
      linkEls.set(e.id, path);
    });

    const nodeEls = new Map();
    nodes.forEach((n) => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "prg-node");
      g.dataset.id = n.id;
      g.dataset.status = n.status;
      g.dataset.theme = n.theme;

      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("class", "prg-node-ring");
      ring.setAttribute("r", "14");
      ring.setAttribute("cx", "0");
      ring.setAttribute("cy", "0");

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("class", "prg-node-circle");
      circle.setAttribute("r", "9");
      circle.setAttribute("cx", "0");
      circle.setAttribute("cy", "0");
      circle.setAttribute("fill", STATUS_COLORS[n.status] || STATUS_COLORS.closed);
      if (n.status === "draft" || n.status === "closed") {
        circle.setAttribute("fill", n.status === "draft" ? "#fff" : "#fff");
        circle.setAttribute("stroke", STATUS_COLORS[n.status]);
        circle.setAttribute("stroke-width", n.status === "draft" ? "2" : "2");
        circle.setAttribute("stroke-dasharray", n.status === "draft" ? "2 2" : "");
      }

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "prg-node-label");
      label.setAttribute("y", "22");
      label.textContent = "PR-" + n.number;

      g.appendChild(ring);
      g.appendChild(circle);
      g.appendChild(label);
      nodeLayer.appendChild(g);
      nodeEls.set(n.id, g);
    });

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
      const angle = (Math.PI * 2 * i) / nodes.length;
      n.x = dims.width / 2 + Math.cos(angle) * Math.min(dims.width, dims.height) * 0.22;
      n.y = dims.height / 2 + Math.sin(angle) * Math.min(dims.width, dims.height) * 0.22;
    });

    const themes = [...new Set(nodes.map((n) => n.theme))];
    const sim = createSimulation(nodes, edges, {
      width: dims.width,
      height: dims.height,
      themeCenters: themeCentersFor(themes, dims.width, dims.height),
      clusterStrength: 0,
    });

    function inTimeRange(n) {
      if (tMax <= tMin) return true;
      const span = tMax - tMin;
      const lo = tMin + ui.timeStart * span;
      const hi = tMin + ui.timeEnd * span;
      return n.t >= lo && n.t <= hi;
    }

    function statusVisible(n) {
      return ui.statuses.has(n.status);
    }

    function searchMatch(n) {
      if (!ui.search) return true;
      const q = ui.search.toLowerCase();
      return (
        String(n.number).includes(q) ||
        n.id.toLowerCase().includes(q) ||
        (n.title || "").toLowerCase().includes(q) ||
        (n.themeLabel || "").toLowerCase().includes(q) ||
        (n.author || "").toLowerCase().includes(q)
      );
    }

    function nodeActive(n) {
      return statusVisible(n) && inTimeRange(n) && searchMatch(n);
    }

    function edgeVisible(e) {
      if (ui.edgeType !== "all" && e.type !== ui.edgeType) return false;
      return nodeActive(e.source) && nodeActive(e.target);
    }

    function applyTransform() {
      const { x, y, k } = ui.transform;
      viewport.setAttribute("transform", "translate(" + x + "," + y + ") scale(" + k + ")");
      $("#prg-zoom-pct", root).textContent = Math.round(k * 100) + "%";
    }

    function renderList() {
      const visible = nodes.filter(nodeActive).sort((a, b) => b.number - a.number);
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
          '<span class="prg-status-dot" data-status="' +
          n.status +
          '"></span><span><div class="prg-node-id">' +
          n.id +
          "</div><div class=\"prg-node-title\">" +
          truncate(n.title, 72) +
          "</div></span>";
        btn.addEventListener("click", () => selectNode(n.id, true));
        nodeList.appendChild(btn);
      });
    }

    function openDrawer(n) {
      drawer.classList.add("is-open");
      drawer.innerHTML =
        '<button type="button" class="prg-drawer-close" data-action="close-drawer" aria-label="Close">×</button>' +
        '<div class="prg-drawer-id">' +
        n.id +
        "</div>" +
        "<h3>" +
        escapeHtml(n.title) +
        "</h3>" +
        "<dl>" +
        "<dt>Status</dt><dd>" +
        STATUS_LABELS[n.status] +
        "</dd>" +
        "<dt>Theme</dt><dd>" +
        escapeHtml(n.themeLabel || n.theme) +
        "</dd>" +
        "<dt>Author</dt><dd>" +
        escapeHtml(n.author) +
        "</dd>" +
        "<dt>Created</dt><dd>" +
        formatDate(n.createdAt) +
        "</dd>" +
        "<dt>Merged</dt><dd>" +
        formatDate(n.mergedAt) +
        "</dd>" +
        "</dl>" +
        (n.excerpt
          ? '<p class="prg-drawer-excerpt">' + escapeHtml(n.excerpt) + "</p>"
          : "") +
        '<a class="prg-open-pr" href="' +
        escapeAttr(n.url) +
        '" target="_blank" rel="noopener noreferrer">Open pull request ↗</a>';
    }

    function closeDrawer() {
      drawer.classList.remove("is-open");
      drawer.innerHTML = "";
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function escapeAttr(s) {
      return escapeHtml(s).replace(/'/g, "&#39;");
    }

    function selectNode(id, panTo) {
      ui.selectedId = id;
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
      // Scroll list item into view
      const item = nodeList.querySelector('[data-id="' + id + '"]');
      if (item && item.scrollIntoView) {
        item.scrollIntoView({ block: "nearest" });
      }
    }

    function paint() {
      const selected = ui.selectedId ? byId[ui.selectedId] : null;
      const neighborIds = new Set();
      if (selected) {
        edges.forEach((e) => {
          if (!edgeVisible(e)) return;
          if (e.source.id === selected.id) neighborIds.add(e.target.id);
          if (e.target.id === selected.id) neighborIds.add(e.source.id);
        });
        neighborIds.add(selected.id);
      }

      edges.forEach((e) => {
        const el = linkEls.get(e.id);
        if (!el) return;
        const vis = edgeVisible(e);
        el.style.display = vis ? "" : "none";
        el.setAttribute("x1", e.source.x);
        el.setAttribute("y1", e.source.y);
        el.setAttribute("x2", e.target.x);
        el.setAttribute("y2", e.target.y);
        el.classList.toggle("is-dimmed", !!(selected && vis && !neighborIds.has(e.source.id)));
        el.classList.toggle(
          "is-hot",
          !!(
            selected &&
            vis &&
            (e.source.id === selected.id || e.target.id === selected.id)
          )
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
          !!(selected && active && !neighborIds.has(n.id))
        );
        // Soft scale fade for near time boundary (animation cue)
        const circle = el.querySelector(".prg-node-circle");
        if (circle && active) {
          const mid = (ui.timeStart + ui.timeEnd) / 2;
          const span = Math.max(ui.timeEnd - ui.timeStart, 0.001);
          const nt = tMax > tMin ? (n.t - tMin) / (tMax - tMin) : 0.5;
          const dist = Math.abs(nt - mid) / span;
          const scale = clamp(1.15 - dist * 0.35, 0.75, 1.15);
          circle.setAttribute("r", String(9 * scale));
        }
      });
    }

    function updateSimParams() {
      sim.state.width = dims.width;
      sim.state.height = dims.height;
      sim.state.themeCenters = themeCentersFor(themes, dims.width, dims.height);
      sim.state.clusterStrength = ui.groupByTheme ? ui.clusterPull * 0.55 : 0;
      sim.state.linkStrength = ui.edgeType === "relates" ? 0.28 : 0.38;
      // Only simulate forces on currently visible edges for clarity
      const activeEdges = edges.filter(edgeVisible);
      sim.state.edges = activeEdges;
      const activeNodes = nodes.filter(nodeActive);
      sim.state.nodes = activeNodes.length ? activeNodes : nodes;
      sim.reheat(0.35);
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

    // Node list initial
    renderList();
    applyTransform();
    kick();

    // Interactions
    $all(".prg-pill", root).forEach((btn) => {
      btn.addEventListener("click", () => {
        $all(".prg-pill", root).forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        ui.edgeType = btn.dataset.edge;
        kick();
        paint();
      });
    });

    $all(".prg-chip", root).forEach((btn) => {
      btn.addEventListener("click", () => {
        const st = btn.dataset.status;
        if (ui.statuses.has(st) && ui.statuses.size === 1) return;
        if (ui.statuses.has(st)) ui.statuses.delete(st);
        else ui.statuses.add(st);
        btn.classList.toggle("is-active", ui.statuses.has(st));
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
      // Slider opens from the start of history up to v (progress over time)
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

    // Zoom / pan
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
      const pad = 48;
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
            const angle = (Math.PI * 2 * i) / nodes.length;
            n.x = dims.width / 2 + Math.cos(angle) * Math.min(dims.width, dims.height) * 0.22;
            n.y = dims.height / 2 + Math.sin(angle) * Math.min(dims.width, dims.height) * 0.22;
            n.vx = 0;
            n.vy = 0;
          });
          kick();
        }
      });
    });

    // Pointer pan + node drag
    let pan = null;
    let drag = null;
    let lastTap = 0;

    canvasWrap.addEventListener("pointerdown", (ev) => {
      if (ev.target.closest && ev.target.closest(".prg-node")) {
        const g = ev.target.closest(".prg-node");
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

    // Pinch zoom
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
      sim.state.themeCenters = themeCentersFor(themes, dims.width, dims.height);
      sim.reheat(0.2);
      if (!raf) raf = requestAnimationFrame(loop);
    });
    ro.observe(canvasWrap);

    // Mark page for CSS hooks
    document.body.classList.add("pr-graph-page");
    document.documentElement.classList.add("pr-graph-page");
  }

  async function boot() {
    const root = document.getElementById("pr-progress-graph");
    if (!root) return;
    root.innerHTML = '<div class="prg-loading">Loading PR progress graph…</div>';
    try {
      const data = await loadDataSmart();
      if (!data || !Array.isArray(data.nodes)) {
        throw new Error("Invalid PR graph payload");
      }
      buildApp(root, data);
    } catch (err) {
      console.error(err);
      root.innerHTML =
        '<div class="prg-error">Could not load PR graph data. ' +
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
