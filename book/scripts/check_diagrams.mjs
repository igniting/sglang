// Verify every generated figure in a real browser, where text metrics are exact.
//
// scripts/diagrams.py validates layouts against an estimated character width.
// This does the same checks against the browser's own measurements, and writes a
// screenshot of each figure in both themes so they can be looked at.
//
//   node book/scripts/check_diagrams.mjs [outdir]

import { chromium } from "playwright";
import { readdirSync, mkdirSync } from "fs";
import { resolve } from "path";

const OUT = resolve(process.cwd(), "book/output");
const SHOTS = resolve(process.argv[2] ?? "/tmp/figures");
mkdirSync(SHOTS, { recursive: true });

const pages = readdirSync(OUT).filter((f) => f.endsWith(".html"));
const browser = await chromium.launch(
  process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}
);
const problems = [];
let figures = 0;

for (const theme of ["light", "navy"]) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.addInitScript((t) => {
    try {
      localStorage.setItem("mdbook-theme", t);
    } catch {}
  }, theme);

  for (const file of pages) {
    await page.goto(`file://${OUT}/${file}`, { waitUntil: "load" });
    const found = await page.$$("figure svg");
    if (!found.length) continue;

    // Structural audit, run inside the page against real getBBox() metrics.
    const errs = await page.evaluate((themeName) => {
      const out = [];
      for (const svg of document.querySelectorAll("figure svg")) {
        const title = svg.querySelector("title")?.textContent ?? "(untitled)";
        const vb = svg.viewBox.baseVal;
        const rects = [...svg.querySelectorAll("rect")].map((r) => ({
          x: +r.getAttribute("x"),
          y: +r.getAttribute("y"),
          w: +r.getAttribute("width"),
          h: +r.getAttribute("height"),
        }));

        for (const t of svg.querySelectorAll("text")) {
          const b = t.getBBox();
          const txt = (t.textContent ?? "").slice(0, 34);

          if (b.x < 0 || b.y < 0 || b.x + b.width > vb.width ||
              b.y + b.height > vb.height) {
            out.push(`${themeName} · ${title}: "${txt}" escapes the viewBox`);
          }

          // A label is either fully inside a box or fully clear of every box.
          // Straddling an edge is the failure the hand-drawn figures had.
          for (const r of rects) {
            const inside =
              b.x >= r.x && b.y >= r.y &&
              b.x + b.width <= r.x + r.w && b.y + b.height <= r.y + r.h;
            const clear =
              b.x + b.width <= r.x || b.x >= r.x + r.w ||
              b.y + b.height <= r.y || b.y >= r.y + r.h;
            if (!inside && !clear) {
              out.push(
                `${themeName} · ${title}: "${txt}" straddles the edge of a box`
              );
            }
          }
        }

        // Colour must actually resolve: an unset custom property renders black
        // in both themes and silently kills the dark palette.
        const probe = svg.querySelector("text");
        if (probe && getComputedStyle(probe).fill === "rgb(0, 0, 0)") {
          out.push(`${themeName} · ${title}: text fill did not resolve`);
        }
      }
      return out;
    }, theme);
    problems.push(...errs);

    for (let i = 0; i < found.length; i++) {
      const slug = file.replace(/\.html$/, "");
      const name = found.length > 1 ? `${slug}-${i}` : slug;
      await found[i].screenshot({ path: `${SHOTS}/${name}.${theme}.png` });
      if (theme === "light") figures++;
    }
  }
  await page.close();
}

await browser.close();

console.log(`${figures} figures checked in 2 themes; shots in ${SHOTS}`);
if (problems.length) {
  console.log("\nPROBLEMS:");
  for (const p of [...new Set(problems)]) console.log(`  ${p}`);
  process.exit(1);
}
console.log("no overflow, no straddled edges, colours resolve in both themes");
