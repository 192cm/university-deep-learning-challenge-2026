import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const csvPath = process.argv[2];
const previewPath = process.argv[3];
if (!csvPath || !previewPath) {
  throw new Error("usage: verify_submission.mjs <submission.csv> <preview.png>");
}

const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Submission" });
const sheet = workbook.worksheets.getItem("Submission");
const used = sheet.getUsedRange(true);

sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.getRange("A1:B1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
sheet.getRange("A2:A832").format.numberFormat = "@";
sheet.getRange("B2:B832").format.numberFormat = "0";
sheet.getRange("A1:A832").format.columnWidth = 20;
sheet.getRange("B1:B832").format.columnWidth = 16;
sheet.getRange("A1:B32").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#D9E2F3",
};

const head = await workbook.inspect({
  kind: "table",
  sheetId: "Submission",
  range: "A1:B10",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 2,
  maxChars: 5000,
});
const tail = await workbook.inspect({
  kind: "table",
  sheetId: "Submission",
  range: "A823:B832",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 2,
  maxChars: 5000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 20 },
  summary: "submission formula error scan",
});
const preview = await workbook.render({
  sheetName: "Submission",
  range: "A1:B30",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

console.log(JSON.stringify({
  usedRange: used?.address ?? null,
  head: head.ndjson,
  tail: tail.ndjson,
  errors: errors.ndjson,
  previewPath,
}));
