#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const source = path.join(root, "web");
const output = path.join(root, "dist", "browser");
const boardSource = path.join(root, "reference", "map-assets", "saito", "TS Map-11.jpg");
const required = ["index.html", "styles.css", "app.mjs", "ui-core.mjs", "board-layout.mjs"];

for (const file of required) {
  const absolute = path.join(source, file);
  if (!fs.existsSync(absolute)) throw new Error(`missing browser asset: ${file}`);
}

for (const file of ["app.mjs", "ui-core.mjs", "board-layout.mjs"]) {
  execFileSync(process.execPath, ["--check", path.join(source, file)], { stdio: "inherit" });
}

if (!fs.existsSync(boardSource)) throw new Error("missing supplied Saito board asset");
const boardHash = require("node:crypto").createHash("sha256").update(fs.readFileSync(boardSource)).digest("hex");
if (boardHash !== "8f21635165f354214fa0b7329b3f3e9f2c2769be76ba2dafd192252591f53217") {
  throw new Error(`unexpected Saito board SHA-256: ${boardHash}`);
}

fs.rmSync(output, { recursive: true, force: true });
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.cpSync(source, output, { recursive: true });
fs.mkdirSync(path.join(output, "assets"), { recursive: true });
fs.copyFileSync(boardSource, path.join(output, "assets", "saito-board.jpg"));
console.log(`Browser build ready: ${path.relative(root, output)}`);
