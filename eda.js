import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Papa from 'papaparse';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load the CSV file
const csvPath = path.join(__dirname, '../data/raw/diabetic_data.csv');
const csvContent = fs.readFileSync(csvPath, 'utf8');

// Parse CSV
const parsed = Papa.parse(csvContent, {
  header: true,
  skipEmptyLines: true,
  dynamicTyping: false // Keep all as strings initially
});

const data = parsed.data;
const columns = parsed.meta.fields;

console.log('=== EXPLORATORY DATA ANALYSIS ===\n');
console.log(`Total records: ${data.length}`);
console.log(`Total columns: ${columns.length}\n`);

// Column names
console.log('=== COLUMN NAMES ===');
console.log(columns.join(', '));
console.log('\n');

// Check for missing values (represented as '?' in this dataset)
console.log('=== MISSING VALUES ===');
const missingCounts = {};
columns.forEach(col => {
  const missing = data.filter(row => !row[col] || row[col] === '?' || row[col] === '').length;
  if (missing > 0) {
    missingCounts[col] = missing;
  }
});

if (Object.keys(missingCounts).length > 0) {
  Object.entries(missingCounts)
    .sort((a, b) => b[1] - a[1])
    .forEach(([col, count]) => {
      const percentage = ((count / data.length) * 100).toFixed(2);
      console.log(`${col}: ${count} (${percentage}%)`);
    });
} else {
  console.log('No missing values detected.');
}
console.log('\n');

// Label distribution (readmitted column)
console.log('=== READMISSION LABEL DISTRIBUTION ===');
if (columns.includes('readmitted')) {
  const readmittedCounts = {};
  data.forEach(row => {
    const value = row['readmitted'];
    readmittedCounts[value] = (readmittedCounts[value] || 0) + 1;
  });

  Object.entries(readmittedCounts).forEach(([label, count]) => {
    const percentage = ((count / data.length) * 100).toFixed(2);
    console.log(`${label}: ${count} (${percentage}%)`);
  });

  // Binary classification: <30 days vs others
  console.log('\n=== BINARY CLASSIFICATION (READMITTED <30 DAYS) ===');
  const readmittedUnder30 = data.filter(row => row['readmitted'] === '<30').length;
  const notReadmittedUnder30 = data.length - readmittedUnder30;
  console.log(`Readmitted <30 days (positive): ${readmittedUnder30} (${((readmittedUnder30 / data.length) * 100).toFixed(2)}%)`);
  console.log(`Not readmitted <30 days (negative): ${notReadmittedUnder30} (${((notReadmittedUnder30 / data.length) * 100).toFixed(2)}%)`);
} else {
  console.log('readmitted column not found!');
}
console.log('\n');

// Sample data types and unique values for key columns
console.log('=== SAMPLE DATA FOR KEY COLUMNS ===');
const keyColumns = ['age', 'gender', 'time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient', 'readmitted'];
keyColumns.forEach(col => {
  if (columns.includes(col)) {
    const uniqueValues = [...new Set(data.map(row => row[col]))];
    console.log(`${col}: ${uniqueValues.length} unique values`);
    if (uniqueValues.length <= 10) {
      console.log(`  Values: ${uniqueValues.join(', ')}`);
    } else {
      console.log(`  Sample: ${uniqueValues.slice(0, 10).join(', ')}...`);
    }
  }
});

console.log('\n=== EDA COMPLETE ===');
