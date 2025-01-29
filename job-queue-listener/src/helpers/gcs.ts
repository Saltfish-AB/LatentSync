import { Storage } from "@google-cloud/storage";
import { ENVIRONMENT, GCS_BUCKET, SA_SECRET_PATH } from "../configs";
import * as fs from "fs";
import * as path from "path";

// Initialize Google Cloud Storage
const storage = new Storage({
  keyFilename: SA_SECRET_PATH,
});

/**
 * Uploads a JSON object as a file to Google Cloud Storage.
 * @param bucketName - The name of the Google Cloud Storage bucket.
 * @param fileName - The name of the file to upload (including .json extension).
 * @param data - The JSON object to be uploaded.
 * @returns A promise that resolves when the upload is complete.
 */
export const uploadJSONToGCS = async (
  fileName: string,
  data: object
): Promise<void> => {
  try {
    // Convert JSON object to string
    const jsonContent = JSON.stringify(data);

    // Create a temporary file path
    const tempFilePath = path.join(__dirname, `${fileName}.json`);

    // Write the JSON data to a temporary file
    fs.writeFileSync(tempFilePath, jsonContent);

    // Upload the file to Google Cloud Storage
    await storage.bucket(GCS_BUCKET).upload(tempFilePath, {
      destination: fileName,
      contentType: "application/json",
    });

    // Delete the temporary file after upload
    fs.unlinkSync(tempFilePath);

    console.log(`File ${fileName} uploaded to ${GCS_BUCKET}.`);
  } catch (error) {
    console.error("Error uploading JSON to GCS:", error);
    throw error;
  }
};

/**
 * Uploads a JSON file directly to Google Cloud Storage from a file path.
 * @param bucketName - The name of the Google Cloud Storage bucket.
 * @param filePath - The local path to the JSON file.
 * @param destination - The destination file name in the bucket.
 * @returns A promise that resolves when the upload is complete.
 */
export const uploadFileToGCS = async (
  filePath: string,
  destination: string,
  contentType: string = "application/json"
): Promise<void> => {
  try {
    // Upload the file to Google Cloud Storage
    await storage.bucket(GCS_BUCKET).upload(filePath, {
      destination,
      contentType,
    });

    console.log(`File ${destination} uploaded to ${GCS_BUCKET}.`);
  } catch (error) {
    console.error("Error uploading file to GCS:", error);
    throw error;
  }
};
