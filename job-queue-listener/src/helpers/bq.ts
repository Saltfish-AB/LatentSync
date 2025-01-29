import { BigQuery, Table } from "@google-cloud/bigquery";
import { GCP_PROJECT_ID } from "../configs";

// Initialize the BigQuery client
const bigquery = new BigQuery({
  projectId: GCP_PROJECT_ID,
});

// Define a type for the data rows you want to insert into BigQuery
interface Row {
  [key: string]: any; // Each row can have key-value pairs, flexible for BigQuery schema
}

// Helper function to insert rows into a BigQuery table
export async function insertRows(
  datasetId: string,
  tableId: string,
  rows: Row[]
): Promise<{ success: boolean; error?: any }> {
  try {
    // Get the dataset and table reference
    const dataset = bigquery.dataset(datasetId);
    const table: Table = dataset.table(tableId);

    // Insert the rows into the table
    await table.insert(rows);

    return { success: true };
  } catch (error: any) {
    console.error("Error inserting rows:", error);

    if (error.name === "PartialFailureError") {
      error.errors.forEach((err: any) => {
        console.error("BigQuery error:", err);
      });
    }

    return { success: false, error };
  }
}
