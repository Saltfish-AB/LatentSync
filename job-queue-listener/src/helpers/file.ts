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

/**
 * Removes a single file at the specified path.
 * @param filePath - The path to the file to be removed.
 * @returns A Promise that resolves when the file is removed, or rejects with an error.
 */
export async function removeFile(filePath: string): Promise<void> {
  console.log(`Attempting to remove file: ${filePath}`);
  
  try {
    // Check if the file exists before attempting to remove it
    const exists = await fileExists(filePath);
    
    if (!exists) {
      console.warn(`File not found: ${filePath}`);
      return;
    }
    
    // Check if the path is a directory
    const stat = await fs.lstat(filePath);
    if (stat.isDirectory()) {
      throw new Error(`Cannot use removeFile on a directory: ${filePath}. Use removeAllFiles instead.`);
    }
    
    // Remove the file
    await fs.unlink(filePath);
    console.log(`File successfully removed: ${filePath}`);
  } catch (error) {
    console.error(`Error removing file: ${filePath}`, error);
    throw error; // Re-throw the error for the caller to handle
  }
}