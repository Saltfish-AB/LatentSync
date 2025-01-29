import fs from "fs";
import https from "https";
import http from "http";

/**
 * Downloads a file from a URL and saves it to a specified local path.
 * @param fileUrl - The URL of the file to download.
 * @param outputPath - The local path where the file should be saved.
 * @returns A Promise that resolves when the download is complete.
 */
export const downloadFile = async (
  fileUrl: string,
  outputPath: string
): Promise<void> => {
  return new Promise((resolve, reject) => {
    // Determine the protocol (http or https)
    const protocol = fileUrl.startsWith("https") ? https : http;

    const file = fs.createWriteStream(outputPath);
    const request = protocol.get(fileUrl, (response) => {
      if (response.statusCode !== 200) {
        reject(
          new Error(`Failed to get '${fileUrl}' (${response.statusCode})`)
        );
        return;
      }

      response.pipe(file);

      file.on("finish", () => {
        file.close(); // Close the file stream after writing is complete
        console.log(`File downloaded successfully to ${outputPath}`);
        resolve();
      });
    });

    request.on("error", (err) => {
      fs.unlink(outputPath, () => reject(err)); // Delete the file on error
    });

    file.on("error", (err) => {
      fs.unlink(outputPath, () => reject(err)); // Delete the file on error
    });
  });
};
