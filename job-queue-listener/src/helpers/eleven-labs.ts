import fs from 'fs';
import { pipeline } from 'stream';
import { promisify } from 'util';
import { Readable } from 'stream';

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

export type VoiceSettings = {
  stability: number;
  similarity_boost: number;
  style: number;
  use_speaker_boost: boolean;
  speed: number;
}

/**
 * Converts text to speech using ElevenLabs API and saves the audio file locally.
 * @param voiceId - The ID of the voice to use.
 * @param textPrompt - The text to be converted to speech.
 * @param outputFilePath - The path where the audio file should be saved.
 * @param nextText - Optional text that comes after the current text.
 * @param settings - Optional voice settings to customize the speech output.
 */
export const textToSpeech = async (voiceId: string, textPrompt: string, outputFilePath: string, nextText?: string, settings?: VoiceSettings): Promise<void> => {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY is not set in environment variables.');
  }

  try {
    // Prepare the request payload
    const payload: any = {
      text: textPrompt,
      model_id: 'eleven_multilingual_v2',
    };
    
    // Add nextText if provided
    if (nextText) {
      payload.nextText = nextText;
    }
    
    // Add voice settings if provided
    if (settings) {
      payload.voice_settings = {
        stability: settings.stability,
        similarity_boost: settings.similarity_boost,
        style: settings.style,
        use_speaker_boost: settings.use_speaker_boost,
        speed: settings.speed
      };
    }

    const response = await fetch(`${ELEVENLABS_BASE_URL}/text-to-speech/${voiceId}?output_format=mp3_44100_128`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const responseText = await response.text();
      console.log('Response text:', responseText);
      throw new Error(`Failed to generate speech: ${response.status} ${response.statusText}`);
    }

    // Get the response as an ArrayBuffer
    const arrayBuffer = await response.arrayBuffer();
    
    // Create a Node.js Readable stream from the ArrayBuffer
    const readableStream = new Readable();
    readableStream.push(Buffer.from(arrayBuffer));
    readableStream.push(null); // Signals the end of the stream
    
    // Create the write stream
    const fileStream = fs.createWriteStream(outputFilePath);
    
    // Pipe the readable stream to the file stream
    await pipelineAsync(readableStream, fileStream);

    console.log(`✅ Audio saved to: ${outputFilePath}`);
  } catch (error) {
    console.error('Error in text-to-speech:', error);
    throw error;
  }
};