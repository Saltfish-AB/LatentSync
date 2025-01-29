import { exec } from "child_process";
import path from "path";
import { promises as fs } from "fs";

// Type definition for silence and speaking segments
interface Segment {
  start: number;
  end: number;
}

/**
 * Finds silent segments in an audio file.
 * @param inputFile - Path to the input audio file.
 * @param minSilenceDuration - Minimum duration for a segment to be considered silence.
 * @returns Promise resolving with an array of silent segments.
 */
function findSilentSegments(
  inputFile: string,
  minSilenceDuration = 0.5
): Promise<Segment[]> {
  return new Promise((resolve, reject) => {
    console.log(`Running findSilentSegments on file: ${inputFile}`);
    const command = `ffmpeg -y -i ${inputFile} -af silencedetect=n=-30dB:d=0.5 -f null -`;
    console.log(`Executing command: ${command}`);

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error during findSilentSegments: ${error.message}`);
        reject(`Error executing FFmpeg: ${error.message}`);
        return;
      }

      const silenceData: Segment[] = [];
      const regex = /silence_(start|end): ([0-9.]+)/g;
      let match;

      while ((match = regex.exec(stderr)) !== null) {
        const type = match[1];
        const time = parseFloat(match[2]);

        if (type === "start") {
          silenceData.push({ start: time, end: 0 });
        } else if (type === "end") {
          const last = silenceData[silenceData.length - 1];
          if (last && !last.end) {
            last.end = time;

            // Filter silence segments based on duration
            const duration = last.end - last.start;
            if (duration < minSilenceDuration) {
              silenceData.pop(); // Remove segment if it's too short
            }
          }
        }
      }

      console.log("Detected silence segments:", silenceData);
      resolve(silenceData);
    });
  });
}

/**
 * Gets the duration of an audio file.
 * @param inputFile - Path to the input audio file.
 * @returns Promise resolving with the duration in seconds.
 */
function getAudioDuration(inputFile: string): Promise<number> {
  return new Promise((resolve, reject) => {
    console.log(`Running getAudioDuration on file: ${inputFile}`);
    const command = `ffprobe -i ${inputFile} -show_entries format=duration -v quiet -of csv="p=0"`;
    console.log(`Executing command: ${command}`);

    exec(command, (error, stdout) => {
      if (error) {
        console.error(`Error during getAudioDuration: ${error.message}`);
        reject(`Error fetching duration: ${error.message}`);
        return;
      }

      const duration = parseFloat(stdout.trim());
      console.log(`Audio duration: ${duration} seconds`);
      resolve(duration);
    });
  });
}

/**
 * Calculates speaking segments from silent segments and audio duration.
 * @param silenceSegments - Array of silent segments.
 * @param duration - Total duration of the audio file.
 * @returns Array of speaking segments.
 */
function calculateSpeakingSegments(
  silenceSegments: Segment[],
  duration: number
): Segment[] {
  console.log("Calculating speaking segments...");
  console.log("Input silence segments:", silenceSegments);
  console.log("Total audio duration:", duration);

  const speakingSegments: Segment[] = [];
  let lastEnd = 0; // Tracks the end of the last silence

  silenceSegments.forEach(({ start, end }) => {
    if (lastEnd < start) {
      speakingSegments.push({ start: lastEnd, end: start });
    }
    lastEnd = end; // Update the end of the last silence
  });

  // Check if there's speaking after the last silence
  if (lastEnd < duration) {
    speakingSegments.push({ start: lastEnd, end: duration });
  }

  console.log("Calculated speaking segments:", speakingSegments);
  return speakingSegments;
}

/**
 * Copies the input audio file to the output file.
 * @param inputFile - Path to the input audio file.
 * @param outputFile - Path to the output audio file.
 * @returns Promise resolving when the copy is complete.
 */
function copyAudioFile(inputFile: string, outputFile: string): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`Copying file from ${inputFile} to ${outputFile}`);
    const command = `ffmpeg -y -i ${inputFile} -c copy ${outputFile}`;
    console.log(`Executing command: ${command}`);

    exec(command, (error) => {
      if (error) {
        console.error(`Error during copyAudioFile: ${error.message}`);
        reject(`Error copying audio file: ${error.message}`);
        return;
      }
      console.log("Input audio file copied to output:", outputFile);
      resolve();
    });
  });
}

/**
 * Finds the segment with a duration closest to the target duration.
 * @param segments - Array of segments with start and end times.
 * @param targetDuration - The target duration to compare against.
 * @returns The segment closest to the target duration.
 * @throws If the segments array is empty or undefined.
 */
function findClosestSegment(
  segments: Segment[],
  targetDuration: number
): Segment | null {
  if (!segments || segments.length === 0) {
    throw new Error("Segments array is empty or undefined.");
  }

  let closestSegment: Segment | null = null;
  let smallestDifference: number = Infinity;

  segments.forEach((segment) => {
    const duration = segment.end - segment.start;
    const difference = Math.abs(duration - targetDuration);

    if (difference < smallestDifference) {
      smallestDifference = difference;
      closestSegment = segment;
    }
  });

  return closestSegment;
}

/**
 * Processes an audio file, extracts the speaking segment closest to a target duration,
 * and creates a new audio clip. If no segments are found, copies the input file to the output file.
 * @param inputFile - Path to the input audio file.
 * @param targetDuration - Target duration for the speaking segment.
 * @param outputFile - Path to the output audio file.
 * @returns Promise resolving when processing is complete.
 */
export async function getShortestAudioSegment(
  inputFile: string,
  targetDuration: number,
  outputFile: string
): Promise<void> {
  console.log(`Processing audio file: ${inputFile}`);
  console.log(`Target duration: ${targetDuration} seconds`);
  console.log(`Output file: ${outputFile}`);

  try {
    // Find silent segments and audio duration concurrently
    const [silenceSegments, audioDuration] = await Promise.all([
      findSilentSegments(inputFile),
      getAudioDuration(inputFile),
    ]);

    console.log("Silence segments:", silenceSegments);
    console.log("Audio duration:", audioDuration);

    // Calculate speaking segments and find the closest one to the target duration
    const speakingSegments = calculateSpeakingSegments(
      silenceSegments,
      audioDuration
    );
    console.log("Speaking segments:", speakingSegments);

    const closestSegment = findClosestSegment(speakingSegments, targetDuration);
    if (!closestSegment) {
      throw new Error("No speaking segments found.");
    }
    if (closestSegment.end - closestSegment.start < 8) {
      console.log("Closest segment is too short, copying the input file...");
      await copyAudioFile(inputFile, outputFile);
      return;
    }
    if (audioDuration < targetDuration) {
      console.log("Audio duration is less than target duration.");
      await copyAudioFile(inputFile, outputFile);
      return;
    }
    await createAudioClipWithSilence(inputFile, outputFile, closestSegment);
  } catch (error) {
    console.error("Error processing audio segment:", error);
    console.log("Falling back to copying the input file...");
    await copyAudioFile(inputFile, outputFile); // Fallback to copying the input file
  }
}

/**
 * Adds 1 second of silence to the end of an audio file.
 * @param inputFile - Path to the input audio file.
 * @param outputFile - Path to the output audio file.
 * @returns Promise resolving when the audio file with silence is created.
 */
export function addSilenceToAudio(
  inputFile: string,
  outputFile: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const command = `ffmpeg -y -i "${inputFile}" -f lavfi -t 1 -i anullsrc=r=44100:cl=stereo -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" -c:a libmp3lame -ar 44100 -b:a 192k "${outputFile}"`;
    console.log(`Executing command: ${command}`);

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error adding silence: ${error.message}`);
        reject(`Error adding silence: ${error.message}`);
        return;
      }

      console.log(`Successfully added 1 second of silence to: ${outputFile}`);
      resolve();
    });
  });
}

/**
 * Executes a shell command and returns a promise that resolves when the command completes.
 * @param command - The shell command to execute.
 * @returns A Promise that resolves when the command finishes.
 */
function executeCommand(command: string): Promise<void> {
  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Command failed: ${stderr}`);
        reject(error);
      } else {
        console.log(stdout);
        resolve();
      }
    });
  });
}

/**
 * Cuts an audio clip based on the given segment, adds 1 second of silence at the end,
 * and saves it as a new file.
 * @param inputFilePath - Path to the input audio file.
 * @param outputFilePath - Path to save the new audio file.
 * @param segment - The segment containing start and end times.
 * @returns A Promise that resolves when the process is complete.
 */
export async function createAudioClipWithSilence(
  inputFilePath: string,
  outputFilePath: string,
  segment: Segment
): Promise<void> {
  const tempFile = path.resolve("temp_audio_clip.mp3");

  try {
    // Step 1: Cut the audio segment with proper encoding
    const cutCommand = `ffmpeg -y -i "${inputFilePath}" -ss ${segment.start} -t ${
      segment.end - segment.start
    } -c:a libmp3lame -ar 44100 -b:a 192k -ac 2 "${tempFile}"`;
    console.log(`Executing: ${cutCommand}`);
    await executeCommand(cutCommand);

    // Step 2: Append 1 second of silence with proper encoding
    const appendCommand = `ffmpeg -y -i "${tempFile}" -f lavfi -t 1 -i anullsrc=r=44100:cl=stereo -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" -c:a libmp3lame -ar 44100 -b:a 192k "${outputFilePath}"
`;
    console.log(`Executing: ${appendCommand}`);
    await executeCommand(appendCommand);

    // Step 3: Clean up temporary file
    await fs.unlink(tempFile);
    console.log(`Temporary file removed: ${tempFile}`);
  } catch (err) {
    console.error("Error during audio processing:", err);
    throw err;
  }
}
