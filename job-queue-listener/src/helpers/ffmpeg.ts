import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import fs from 'fs';

/**
 * Concatenates two video files into a new output file.
 * @param videoUrl1 - First video URL or file path.
 * @param videoUrl2 - Second video URL or file path.
 * @param outputFilePath - The output file path where the concatenated video will be saved.
 * @returns A Promise that resolves when the process is complete.
 */
export const concatVideos = async (
  videoUrl1: string,
  videoUrl2: string,
  outputFilePath: string
): Promise<void> => {
  if (!videoUrl1 || !videoUrl2) {
    throw new Error(`Invalid input files: videoUrl1=${videoUrl1}, videoUrl2=${videoUrl2}`);
  }

  console.log(`🎥 Concatenating videos:\n1️⃣ ${videoUrl1}\n2️⃣ ${videoUrl2}\n🔽 Output: ${outputFilePath}`);

  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(videoUrl1)
      .input(videoUrl2)
      .on('start', (command) => console.log(`🎬 FFmpeg command: ${command}`))
      .on('error', (err) => {
        console.error('❌ FFmpeg Error:', err);
        reject(err);
      })
      .on('end', () => {
        console.log(`✅ Video concatenation complete: ${outputFilePath}`);
        resolve();
      })
      .mergeToFile(outputFilePath, path.resolve(path.dirname(outputFilePath), 'temp'));
  });
};

