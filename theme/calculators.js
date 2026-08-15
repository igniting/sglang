// Interactive figures for the book's two central formulas.
//
// The chapters derive `B* = sπ/2β` and `bytes_per_token = 2·L·H_kv·d·s`, then work them
// through for one model on one GPU. A reader with different hardware has to redo the
// arithmetic by hand, which is exactly the friction that makes a formula feel like a fact
// about someone else's machine rather than a tool.
//
// Markup is declarative: `<div class="bk-calc" data-calc="roofline"></div>`. Each entry in
// CALCS declares its inputs and a compute function returning rows of output plus a verdict
// sentence. No dependencies, no network, and nothing to fail if scripting is off — the
// prose around each one always states the worked example itself.

(function () {
  "use strict";

  // Presentation lives here rather than in custom.css so that the markup and the
  // styling for it are one artifact. Split across two files they are cached
  // independently, and a browser holding a stale stylesheet against fresh HTML
  // renders the whole thing unstyled — which is exactly what happened once.
  // Palette tokens carry fallbacks so a cold custom.css still looks deliberate.
  var STYLE = [
    ".content .bk-calc {",
    "  margin: 3rem 0;",
    "  padding: 1.8rem 2rem 1.6rem;",
    "  border: 1px solid var(--bk-rule, #ddd);",
    "  border-radius: 8px;",
    "  background: var(--bk-callout-bg, rgba(127,127,127,0.06));",
    "  font-family: var(--bk-sans, system-ui, -apple-system, sans-serif);",
    "}",
    ".content .bk-calc:empty { display: none; }",
    ".content .bk-calc-title {",
    "  font-weight: 600;",
    "  font-size: 1.55rem;",
    "  letter-spacing: 0.01em;",
    "  margin-bottom: 1.4rem;",
    "  color: var(--fg, inherit);",
    "}",
    ".content .bk-calc-inputs {",
    "  display: grid;",
    "  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));",
    "  gap: 1.2rem 1.6rem;",
    "  margin-bottom: 1.6rem;",
    "}",
    // Inputs align on their bottom edge whether the label wraps to one line or two.
    ".content .bk-calc-field {",
    "  display: flex;",
    "  flex-direction: column;",
    "  justify-content: flex-end;",
    "  gap: 0.3rem;",
    "  margin: 0;",
    "}",
    ".content .bk-calc-label {",
    "  font-size: 1.4rem;",
    "  line-height: 1.3;",
    "  color: var(--bk-muted, #666);",
    "}",
    ".content .bk-calc-input {",
    "  font-family: var(--bk-mono, ui-monospace, monospace);",
    "  font-size: 1.4rem;",
    "  width: 100%;",
    "  box-sizing: border-box;",
    "  padding: 0.35rem 0.6rem;",
    "  color: var(--fg, inherit);",
    "  background: var(--bg, transparent);",
    "  border: 1px solid var(--bk-rule, #ddd);",
    "  border-radius: 4px;",
    "}",
    ".content .bk-calc-input:focus {",
    "  outline: 2px solid var(--links, #06c);",
    "  outline-offset: 1px;",
    "}",
    ".content .bk-calc-table {",
    "  width: 100%;",
    "  margin: 0;",
    "  border-collapse: collapse;",
    "  font-size: 1.45rem;",
    "  font-family: inherit;",
    "}",
    ".content .bk-calc-table th,",
    ".content .bk-calc-table td {",
    "  border: none;",
    "  border-top: 1px solid var(--bk-rule, #ddd);",
    "  padding: 0.6rem 0;",
    "  text-align: left;",
    "  vertical-align: baseline;",
    "  background: none;",
    "}",
    ".content .bk-calc-table th {",
    "  font-weight: 400;",
    "  color: var(--fg, inherit);",
    "  width: 40%;",
    "}",
    ".content .bk-calc-table tr { background: none; }",
    ".content .bk-calc-table td.bk-calc-value {",
    "  font-family: var(--bk-mono, ui-monospace, monospace);",
    "  font-weight: 600;",
    "  color: var(--links, #06c);",
    "  white-space: nowrap;",
    "  padding-right: 1.6rem;",
    "}",
    ".content .bk-calc-note {",
    "  color: var(--bk-muted, #666);",
    "  font-size: 1.35rem;",
    "}",
    ".content .bk-calc-verdict {",
    "  margin: 1.4rem 0 0;",
    "  font-size: 1.45rem;",
    "  line-height: 1.55;",
    "  color: var(--fg, inherit);",
    "}",
    "@media only screen and (max-width: 700px) {",
    "  .content .bk-calc-table th { width: auto; }",
    "  .content .bk-calc-note { display: none; }",
    "}",
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("bk-calc-style")) return;
    var el = document.createElement("style");
    el.id = "bk-calc-style";
    el.textContent = STYLE;
    document.head.appendChild(el);
  }

  var GB = 1024 * 1024 * 1024;

  function fmt(x, digits) {
    if (!isFinite(x)) return "—";
    if (digits === undefined) digits = x >= 100 ? 0 : x >= 10 ? 1 : 2;
    return x.toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  var CALCS = {
    // ---- Chapter 1 -----------------------------------------------------
    roofline: {
      title: "Critical batch size for your hardware",
      inputs: [
        { k: "peak", label: "Peak tensor-core compute", unit: "TFLOP/s", value: 990,
          min: 10, max: 20000, step: 10 },
        { k: "bw", label: "Memory bandwidth", unit: "TB/s", value: 3.35,
          min: 0.1, max: 30, step: 0.05 },
        { k: "bytes", label: "Bytes per weight element", unit: "",
          value: 2, options: [[2, "BF16 / FP16 — 2"], [1, "FP8 — 1"], [0.5, "FP4 / INT4 — 0.5"]] },
      ],
      compute: function (v) {
        var ridge = (v.peak * 1e12) / (v.bw * 1e12);
        var bstar = (v.bytes * ridge) / 2;
        return {
          rows: [
            ["Ridge point <em>I</em>*", fmt(ridge) + " FLOP/byte",
             "where memory-bound becomes compute-bound"],
            ["Critical batch size <em>B</em>*", fmt(bstar, 0) + " token rows",
             "rows needed to saturate the arithmetic units"],
            ["Cost of a row below <em>B</em>*", "≈ free",
             "the bandwidth was already paid for"],
          ],
          verdict:
            "A batch of " + fmt(bstar, 0) + " rows is the target. Below it the accelerator " +
            "is a very expensive memory controller; above it, arithmetic starts to cost " +
            "real time.",
        };
      },
    },

    // ---- Chapter 9 -----------------------------------------------------
    capacity: {
      title: "How many users will this GPU serve?",
      inputs: [
        { k: "params", label: "Model parameters", unit: "B", value: 70,
          min: 0.5, max: 2000, step: 0.5 },
        { k: "wbytes", label: "Bytes per weight", unit: "", value: 2,
          options: [[2, "BF16 — 2"], [1, "FP8 — 1"], [0.5, "INT4 / FP4 — 0.5"]] },
        { k: "layers", label: "Layers", unit: "", value: 80, min: 1, max: 200, step: 1 },
        { k: "kvheads", label: "KV heads", unit: "", value: 8, min: 1, max: 128, step: 1 },
        { k: "headdim", label: "Head dimension", unit: "", value: 128,
          min: 32, max: 512, step: 8 },
        { k: "kvbytes", label: "Bytes per KV element", unit: "", value: 2,
          options: [[2, "BF16 — 2"], [1, "FP8 — 1"]] },
        { k: "mem", label: "Memory per GPU", unit: "GB", value: 80,
          min: 8, max: 400, step: 1 },
        { k: "bw", label: "Memory bandwidth", unit: "TB/s", value: 3.35,
          min: 0.1, max: 30, step: 0.05 },
        { k: "tp", label: "Tensor-parallel size", unit: "", value: 4,
          options: [[1, "1"], [2, "2"], [4, "4"], [8, "8"], [16, "16"]] },
        { k: "ctx", label: "Average context length", unit: "tokens", value: 4000,
          min: 128, max: 200000, step: 128 },
        { k: "frac", label: "Fraction of memory the pool gets", unit: "", value: 0.88,
          min: 0.5, max: 0.97, step: 0.01 },
      ],
      compute: function (v) {
        var weightsPerGpu = (v.params * 1e9 * v.wbytes) / v.tp;
        // KV heads shard across TP only until they run out, then replicate.
        var headsPerRank = Math.max(1, v.kvheads / v.tp);
        var perToken = 2 * v.layers * headsPerRank * v.headdim * v.kvbytes;
        var perSeq = perToken * v.ctx;
        var budget = v.mem * GB * v.frac - weightsPerGpu;
        var seqs = budget / perSeq;
        var tps = (v.bw * 1e12) / ((v.params * 1e9 * v.wbytes) / v.tp);
        var replicated = v.kvheads < v.tp;
        return {
          rows: [
            ["Weights per GPU", fmt(weightsPerGpu / GB) + " GB",
             v.tp > 1 ? "sharded " + v.tp + " ways" : "unsharded"],
            ["KV per token", fmt(perToken / 1024) + " KB",
             replicated
               ? "KV heads replicated — TP is not shrinking this"
               : "on this rank, after sharding"],
            ["KV per sequence", fmt(perSeq / GB, 2) + " GB",
             "at " + fmt(v.ctx, 0) + " tokens"],
            ["Left for the KV pool", fmt(budget / GB) + " GB",
             "after weights, graphs, and activations"],
            ["Concurrent sequences", budget > 0 ? fmt(Math.floor(seqs), 0) : "0",
             "this is the number that sets throughput"],
            ["Single-stream decode", fmt(tps, 0) + " tok/s",
             "bandwidth ÷ weight bytes — the speed-of-light figure"],
          ],
          verdict:
            budget <= 0
              ? "The weights do not leave room for a KV cache. Raise the parallel size, " +
                "quantize the weights, or use a larger accelerator."
              : "About " + fmt(Math.floor(seqs), 0) + " concurrent conversations across " +
                (v.tp > 1 ? "the " + v.tp + "-GPU deployment. " : "the deployment. ") +
                (replicated
                  ? "KV heads are replicated at this parallel size, so raising it no longer " +
                    "buys cache capacity — this is the case for data-parallel attention."
                  : "Halving KV bytes, or sharing prefixes between requests, moves this " +
                    "number more than any kernel will."),
        };
      },
    },
  };

  function build(root) {
    var spec = CALCS[root.getAttribute("data-calc")];
    if (!spec) return;

    var state = {};
    var form = document.createElement("div");
    form.className = "bk-calc-inputs";

    var head = document.createElement("div");
    head.className = "bk-calc-title";
    head.textContent = spec.title;
    root.appendChild(head);

    spec.inputs.forEach(function (inp) {
      state[inp.k] = inp.value;
      var wrap = document.createElement("label");
      wrap.className = "bk-calc-field";

      var name = document.createElement("span");
      name.className = "bk-calc-label";
      name.textContent = inp.unit ? inp.label + " (" + inp.unit + ")" : inp.label;
      wrap.appendChild(name);

      var el;
      if (inp.options) {
        el = document.createElement("select");
        inp.options.forEach(function (o) {
          var opt = document.createElement("option");
          opt.value = String(o[0]);
          opt.textContent = o[1];
          if (o[0] === inp.value) opt.selected = true;
          el.appendChild(opt);
        });
      } else {
        el = document.createElement("input");
        el.type = "number";
        el.value = String(inp.value);
        el.min = String(inp.min);
        el.max = String(inp.max);
        el.step = String(inp.step);
      }
      el.className = "bk-calc-input";
      el.addEventListener("input", function () {
        var n = parseFloat(el.value);
        state[inp.k] = isNaN(n) ? inp.value : n;
        render();
      });
      wrap.appendChild(el);
      form.appendChild(wrap);
    });
    root.appendChild(form);

    var out = document.createElement("div");
    out.className = "bk-calc-out";
    root.appendChild(out);

    function render() {
      var r = spec.compute(state);
      var html = "<table class='bk-calc-table'><tbody>";
      r.rows.forEach(function (row) {
        html +=
          "<tr><th scope='row'>" + row[0] + "</th><td class='bk-calc-value'>" +
          row[1] + "</td><td class='bk-calc-note'>" + row[2] + "</td></tr>";
      });
      html += "</tbody></table><p class='bk-calc-verdict'>" + r.verdict + "</p>";
      out.innerHTML = html;
    }
    render();
  }

  function init() {
    var nodes = document.querySelectorAll(".bk-calc");
    if (!nodes.length) return;
    injectStyle();
    for (var i = 0; i < nodes.length; i++) build(nodes[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
