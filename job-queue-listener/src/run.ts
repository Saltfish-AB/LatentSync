import fs from "fs";
import { spawn } from "child_process";
import { REMOTE_MODELS_PATH } from "./configs";

export const runScript = (scriptId: string, command: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const logFilePath = `${REMOTE_MODELS_PATH}/logs/${scriptId}.txt`;

    // Remove the log file if it already exists
    if (fs.existsSync(logFilePath)) {
      try {
        fs.unlinkSync(logFilePath);
        console.log(`Existing log file removed: ${logFilePath}`);
      } catch (error) {
        if (error instanceof Error) {
          console.error(`Failed to remove existing log file: ${error.message}`);
        } else {
          console.error(`Failed to remove existing log file: ${error}`);
        }
        return reject(new Error(`Failed to remove existing log file`));
      }
    }

    // Ensure the logs directory exists
    const logsDir = `${REMOTE_MODELS_PATH}/logs`;
    if (!fs.existsSync(logsDir)) {
      try {
        fs.mkdirSync(logsDir, { recursive: true });
        console.log(`Created logs directory: ${logsDir}`);
      } catch (error) {
        if (error instanceof Error) {
          console.error(`Failed to create logs directory: ${error.message}`);
        } else {
          console.error(`Failed to create logs directory: ${error}`);
        }
        return reject(new Error(`Failed to create logs directory`));
      }
    }

    const logStream = fs.createWriteStream(logFilePath, { flags: "a" });

    console.log(`Starting process with command: ${command}`);
    const process = spawn(command, { shell: true });

    // Capture stdout
    process.stdout.on("data", (data: Buffer) => {
      const log = data.toString().trim();
      console.log(`Output: ${log}`);
      logStream.write(log + "\n");
    });

    // Capture stderr
    process.stderr.on("data", (data: Buffer) => {
      const log = `Error: ${data.toString().trim()}`;
      console.error(log);
      logStream.write(log + "\n");
    });

    // Handle process exit
    process.on("close", (code: number) => {
      logStream.end();
      if (code === 0) {
        console.log(`Process exited successfully with code ${code}`);
        resolve();
      } else {
        console.error(`Process exited with code ${code}`);
        reject(new Error(`Process exited with code ${code}`));
      }
    });

    // Handle spawn errors
    process.on("error", (err) => {
      logStream.end();
      console.error(`Failed to start process: ${err.message}`);
      reject(new Error(`Failed to start process: ${err.message}`));
    });
  });
};
