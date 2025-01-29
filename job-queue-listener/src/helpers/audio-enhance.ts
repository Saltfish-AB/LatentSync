import { exec } from "child_process";
import * as fs from "fs";
import * as path from "path";

/**
 * Executes a shell command and returns a Promise.
 * @param command - The command to execute.
 * @returns A Promise that resolves when the command completes.
 */
function execCommand(command: string): Promise<void> {
  return new Promise((resolve, reject) => {
    exec(command, (err, stdout, stderr) => {
      if (err) {
        reject(stderr || err.message);
        return;
      }
      resolve();
    });
  });
}

/**
 * Apply all audio enhancements: change sample rate, normalize audio, reduce noise, apply equalizer, and convert to WAV.
 * @param inputFile - Path to the input audio file.
 * @param outputFile - Path to the output audio file.
 * @param options - Options for audio enhancements.
 * @returns A Promise that resolves when all enhancements are applied.
 */
export async function enhanceAudio(
  inputFile: string,
  outputFile: string,
  options: {
    targetSampleRate?: number;
    frequency?: number;
    gain?: number;
  } = {}
): Promise<void> {
  if (!fs.existsSync(inputFile)) {
    throw new Error(`Input file not found: ${inputFile}`);
  }

  const tempDir = fs.mkdtempSync(path.join(path.dirname(outputFile), "temp-"));
  const tempFiles: string[] = [];

  const {
    targetSampleRate = 22050, // Default to 44100 Hz
    frequency = 100, // Default frequency for equalizer
    gain = 0, // Default gain for equalizer
  } = options;

  // Create temp files for intermediate steps
  const tempSampleRateFile = path.join(tempDir, "temp_sample_rate.mp3");
  const tempNormalizedFile = path.join(tempDir, "temp_normalized.mp3");
  const tempNoiseReducedFile = path.join(tempDir, "temp_noise_reduced.mp3");
  const finalOutputFile = path.join(tempDir, "final_output.mp3");
  tempFiles.push(
    tempSampleRateFile,
    tempNormalizedFile,
    tempNoiseReducedFile,
    finalOutputFile
  );

  const wavOutputFile = outputFile.endsWith(".wav")
    ? outputFile
    : outputFile.replace(path.extname(outputFile), ".wav");

  // Define commands for each step
  const commands = [
    // Step 1: Change sample rate
    `ffmpeg -y -i "${inputFile}" -ar ${targetSampleRate} "${tempSampleRateFile}"`,
    // Step 2: Normalize audio
    //`ffmpeg -y -i "${tempSampleRateFile}" -af "volume=1.0" "${tempNormalizedFile}"`,
    // Step 3: Reduce noise with gentler settings
    //`ffmpeg -y -i "${tempNormalizedFile}" -af "afftdn=nf=-20" "${tempNoiseReducedFile}"`,
    // Step 4: Apply equalizer with subtler settings
    //`ffmpeg -y -i "${tempNoiseReducedFile}" -af "equalizer=f=1000:t=q:w=1:g=1" "${finalOutputFile}"`,
    // Step 5: Convert to WAV
    `ffmpeg -y -i "${tempSampleRateFile}" "${wavOutputFile}"`,
  ];

  try {
    // Run each command sequentially
    for (const [index, command] of commands.entries()) {
      console.log(`Running step ${index + 1}...`);
      await execCommand(command);
      console.log(`Step ${index + 1} completed successfully.`);
    }

    console.log(
      `Enhancements applied successfully. Output saved to ${wavOutputFile}`
    );
  } catch (error) {
    console.error(`Error during enhancement process: ${error}`);
    throw error;
  } finally {
    // Cleanup temporary files
    tempFiles.forEach((file) => {
      if (fs.existsSync(file)) {
        fs.unlinkSync(file);
      }
    });
    if (fs.existsSync(tempDir)) {
      fs.rmdirSync(tempDir, { recursive: true });
    }
    console.log("Temporary files cleaned up.");
  }
}
