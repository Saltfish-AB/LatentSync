import fs from 'fs';
import { pipeline } from 'stream';
import { promisify } from 'util';

const pipelineAsync = promisify(pipeline);

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1";

/**
 * Fetches details of a specific voice from ElevenLabs API
 * @param voiceId - The ID of the voice to fetch
 * @returns Promise resolving to the voice details JSON
 */
export const getVoiceDetails = async (voiceId: string): Promise<any> => {
  if (!ELEVENLABS_API_KEY) {
    throw new Error("ELEVENLABS_API_KEY is not set in environment variables.");
  }

  try {
    const response = await fetch(`${ELEVENLABS_BASE_URL}/voices/${voiceId}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
      },
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch voice details: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching voice details:", error);
    throw error;
  }
};

/**
 * Converts text to speech using ElevenLabs API and saves the audio file locally.
 * @param voiceId - The ID of the voice to use.
 * @param textPrompt - The text to be converted to speech.
 * @param outputFilePath - The path where the audio file should be saved.
 */
export const textToSpeech = async (voiceId: string, textPrompt: string, outputFilePath: string, nextText?: string): Promise<void> => {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY is not set in environment variables.');
  }

  try {
    const response = await fetch(`${ELEVENLABS_BASE_URL}/text-to-speech/${voiceId}?output_format=mp3_44100_128`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text: textPrompt,
        model_id: 'eleven_multilingual_v2',
        nextText
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to generate speech: ${response.status} ${response.statusText}`);
    }

    const fileStream = fs.createWriteStream(outputFilePath);
    await pipelineAsync(response.body as unknown as NodeJS.ReadableStream, fileStream);

    console.log(`✅ Audio saved to: ${outputFilePath}`);
  } catch (error) {
    console.error('Error in text-to-speech:', error);
    throw error;
  }
};
