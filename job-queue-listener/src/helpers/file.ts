import { promises as fs } from "fs";
import * as path from "path";

/**
 * Checks whether a file exists at the given path.
 * @param path - The file path to check.
 * @returns A boolean indicating if the file exists.
 */
export const fileExists = async (path: string): Promise<boolean> => {
  console.log(`Checking if file exists at path: ${path}`);
  try {
    await fs.access(path);
    return true;
  } catch {
    return false;
  }
};

/**
 * Recursively removes all files and folders in a specified directory.
 * @param directoryPath - The path to the directory to clean.
 */
export async function removeAllFiles(directoryPath: string): Promise<void> {
  try {
    // Check if the directory exists
    await fs.access(directoryPath);

    // Read all files and directories in the given directory
    const items = await fs.readdir(directoryPath);

    for (const item of items) {
      const itemPath = path.join(directoryPath, item);

      // Check if the item is a directory or file
      const stat = await fs.lstat(itemPath);

      if (stat.isDirectory()) {
        // Recursively remove items in the subdirectory
        await removeAllFiles(itemPath);
        // Remove the empty directory
        await fs.rmdir(itemPath);
      } else {
        // Remove the file
        await fs.unlink(itemPath);
      }
    }

    console.log(`All files and subfolders removed from: ${directoryPath}`);
  } catch (error) {
    console.error(
      `Error accessing or processing directory: ${directoryPath}`,
      error
    );
  }
}
