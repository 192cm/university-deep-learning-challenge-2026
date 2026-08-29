import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const INTEGER_RE = /^-?\d+$/;

const artifactToolModule =
  process.env.T12B_ARTIFACT_TOOL_MODULE ?? "@oai/artifact-tool";
const { SpreadsheetFile, Workbook } = await import(artifactToolModule);

function parseArgs(argv) {
  const parsed = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Expected --key value arguments, got ${argv.slice(index).join(" ")}`);
    }
    parsed.set(key.slice(2), value);
  }
  const required = ["rows", "reference", "output", "artifact-dir"];
  for (const key of required) {
    if (!parsed.has(key)) {
      throw new Error(`Missing required argument --${key}`);
    }
  }
  return Object.fromEntries(parsed);
}

function normalizeCell(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function csvCell(value) {
  const text = normalizeCell(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function toCsv(rows) {
  return `${rows.map((row) => row.map(csvCell).join(",")).join("\n")}\n`;
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function assertUnique(values, label) {
  const unique = new Set(values);
  if (unique.size !== values.length) {
    throw new Error(`${label} contains ${values.length - unique.size} duplicate value(s)`);
  }
}

function assertMatrixEqual(actual, expected, label) {
  if (actual.length !== expected.length) {
    throw new Error(`${label} row count mismatch: ${actual.length} != ${expected.length}`);
  }
  for (let rowIndex = 0; rowIndex < expected.length; rowIndex += 1) {
    const actualRow = actual[rowIndex].map(normalizeCell);
    const expectedRow = expected[rowIndex].map(normalizeCell);
    if (actualRow.length !== expectedRow.length) {
      throw new Error(`${label} column count mismatch at row ${rowIndex + 1}`);
    }
    for (let columnIndex = 0; columnIndex < expectedRow.length; columnIndex += 1) {
      if (actualRow[columnIndex] !== expectedRow[columnIndex]) {
        throw new Error(
          `${label} mismatch at row ${rowIndex + 1}, column ${columnIndex + 1}: ` +
            `${JSON.stringify(actualRow[columnIndex])} != ${JSON.stringify(expectedRow[columnIndex])}`,
        );
      }
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const rowsPath = path.resolve(args.rows);
  const referencePath = path.resolve(args.reference);
  const outputPath = path.resolve(args.output);
  const artifactDir = path.resolve(args["artifact-dir"]);
  const backupPath = path.join(artifactDir, "preexisting-submission.csv");
  const candidatePath = path.join(artifactDir, "submission-candidate.csv");
  const xlsxPath = path.join(artifactDir, "submission-verification.xlsx");
  const previewTopPath = path.join(artifactDir, "submission-preview-top.png");
  const previewBottomPath = path.join(artifactDir, "submission-preview-bottom.png");
  const auditPath = path.join(artifactDir, "submission-file-audit.json");

  await fs.mkdir(artifactDir, { recursive: true });

  const payload = JSON.parse(await fs.readFile(rowsPath, "utf8"));
  if (JSON.stringify(payload.columns) !== JSON.stringify(["id", "answer"])) {
    throw new Error(`Unexpected submission columns: ${JSON.stringify(payload.columns)}`);
  }
  if (!Array.isArray(payload.rows) || payload.rows.length === 0) {
    throw new Error("submission-rows.json has no rows");
  }

  const normalizedRows = payload.rows.map((row, index) => {
    const id = normalizeCell(row.id);
    const answer = normalizeCell(row.answer);
    if (!id) throw new Error(`Blank id at submission row ${index + 1}`);
    if (!INTEGER_RE.test(answer)) {
      throw new Error(`Non-canonical integer answer at submission row ${index + 1}: ${answer}`);
    }
    return [id, answer];
  });
  const outputIds = normalizedRows.map(([id]) => id);
  assertUnique(outputIds, "submission ids");

  const referenceCsv = await fs.readFile(referencePath, "utf8");
  const referenceWorkbook = await Workbook.fromCSV(referenceCsv, { sheetName: "Reference" });
  const referenceSheet = referenceWorkbook.worksheets.getItem("Reference");
  const referenceValues = referenceSheet.getUsedRange(true).values;
  if (referenceValues.length !== normalizedRows.length + 1) {
    throw new Error(
      `Reference row count mismatch: ${referenceValues.length - 1} != ${normalizedRows.length}`,
    );
  }
  if (normalizeCell(referenceValues[0][0]) !== "id") {
    throw new Error(`Reference first column is not id: ${normalizeCell(referenceValues[0][0])}`);
  }
  const referenceIds = referenceValues.slice(1).map((row) => normalizeCell(row[0]));
  assertUnique(referenceIds, "reference ids");
  for (let index = 0; index < referenceIds.length; index += 1) {
    if (outputIds[index] !== referenceIds[index]) {
      throw new Error(
        `ID order mismatch at row ${index + 1}: ${outputIds[index]} != ${referenceIds[index]}`,
      );
    }
  }

  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Submission");
  const values = [["id", "answer"], ...normalizedRows];
  const lastRow = values.length;
  sheet.getRange(`A1:B${lastRow}`).values = values;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(`A1:B${lastRow}`).format.numberFormat = "@";
  sheet.getRange(`A1:B${lastRow}`).format.autofitColumns();
  sheet.getRange(`A1:A${lastRow}`).format.columnWidth = 18;
  sheet.getRange(`B1:B${lastRow}`).format.columnWidth = 14;

  const roundTripValues = sheet.getRange(`A1:B${lastRow}`).values;
  assertMatrixEqual(roundTripValues, values, "in-memory workbook");

  const keyInspection = await workbook.inspect({
    kind: "region",
    sheetId: "Submission",
    range: `A1:B${Math.min(lastRow, 15)}`,
    maxChars: 5000,
  });
  const tailInspection = await workbook.inspect({
    kind: "region",
    sheetId: "Submission",
    range: `A${Math.max(1, lastRow - 14)}:B${lastRow}`,
    maxChars: 5000,
  });
  const formulaErrorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
    maxChars: 5000,
  });

  const topPreview = await workbook.render({
    sheetName: "Submission",
    range: `A1:B${Math.min(lastRow, 20)}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(previewTopPath, new Uint8Array(await topPreview.arrayBuffer()));
  const bottomPreview = await workbook.render({
    sheetName: "Submission",
    range: `A${Math.max(1, lastRow - 19)}:B${lastRow}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(previewBottomPath, new Uint8Array(await bottomPreview.arrayBuffer()));

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(xlsxPath);

  const candidateCsv = toCsv(roundTripValues);
  await fs.writeFile(candidatePath, candidateCsv, "utf8");
  const importedWorkbook = await Workbook.fromCSV(candidateCsv, { sheetName: "Submission" });
  const importedValues = importedWorkbook.worksheets
    .getItem("Submission")
    .getRange(`A1:B${lastRow}`).values;
  assertMatrixEqual(importedValues, values, "CSV import round trip");

  let preexistingSha256 = null;
  try {
    const preservedBackup = await fs.readFile(backupPath);
    preexistingSha256 = sha256(preservedBackup);
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
    try {
      const preexisting = await fs.readFile(outputPath);
      preexistingSha256 = sha256(preexisting);
      await fs.writeFile(backupPath, preexisting, { flag: "wx" });
    } catch (outputError) {
      if (outputError?.code !== "ENOENT") throw outputError;
    }
  }
  await fs.writeFile(outputPath, candidateCsv, "utf8");
  const finalBytes = await fs.readFile(outputPath);
  const candidateBytes = await fs.readFile(candidatePath);
  if (!finalBytes.equals(candidateBytes)) {
    throw new Error("Root submission.csv differs from the verified candidate bytes");
  }

  const zeroAnswers = normalizedRows.filter(([, answer]) => answer === "0").length;
  const audit = {
    schema_version: 1,
    task: "T12b-4970-override",
    status: "complete",
    artifact_tool: true,
    rows: normalizedRows.length,
    columns: ["id", "answer"],
    unique_ids: new Set(outputIds).size,
    reference_ids_exact_order_match: true,
    canonical_integer_answers: normalizedRows.length,
    null_or_nan: 0,
    numeric_zero_answers: zeroAnswers,
    forced_zero_fallbacks: 0,
    output_sha256: sha256(finalBytes),
    candidate_sha256: sha256(candidateBytes),
    preexisting_output_sha256: preexistingSha256,
    inputs: {
      submission_rows: rowsPath,
      reference_questions: referencePath,
    },
    outputs: {
      submission: outputPath,
      candidate: candidatePath,
      backup: preexistingSha256 === null ? null : backupPath,
      verification_xlsx: xlsxPath,
      preview_top: previewTopPath,
      preview_bottom: previewBottomPath,
    },
    inspections: {
      key_region: keyInspection.ndjson,
      tail_region: tailInspection.ndjson,
      formula_errors: formulaErrorInspection.ndjson,
    },
  };
  await fs.writeFile(auditPath, `${JSON.stringify(audit, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify({ ...audit, audit: auditPath }, null, 2)}\n`);
}

await main();
