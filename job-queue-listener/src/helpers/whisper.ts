// Install required packages:
// npm install typescript @types/node ts-node --save-dev

import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

// Promisify exec
const execAsync = promisify(exec);

const outputDir: string = path.join(__dirname, 'output');

// Function to run whisper CLI command through Python
async function runWhisperCommand(audioPath: string, outputPath: string): Promise<void> {
  // Ensure output directory exists
  if (!fs.existsSync(outputPath)) {
    fs.mkdirSync(outputPath, { recursive: true });
  }
  
  // Command to run whisper CLI via python -m
  // This is the format for pip-installed Whisper
  const command = `python -m whisper "${audioPath}" --model base --output_dir "${outputPath}" --output_format vtt --word_timestamps True --max_line_width 30 --max_line_count 1 --highlight_words False`;
  
  console.log('Running Whisper command...');
  console.log(command);
  
  try {
    const { stdout, stderr } = await execAsync(command);
    console.log('Whisper output:', stdout);
    if (stderr) {
      console.error('Whisper stderr:', stderr);
    }
  } catch (error) {
    console.error('Error running Whisper command:', error);
    throw error;
  }
}

// Function to modify the VTT file to add line breaks
function addLineBreaksToVTT(vttContent: string): string {
  const lines = vttContent.split('\n');
  const modifiedLines: string[] = [];
  
  let inCue = false;
  let cueText = '';
  
  for (const line of lines) {
    // Check if this is a timestamp line (contains "-->")
    if (line.includes('-->')) {
      inCue = true;
      modifiedLines.push(line);
      continue;
    }
    
    // If we're in a cue and encounter an empty line, we're at the end of a cue
    if (inCue && line.trim() === '') {
      inCue = false;
      
      // Add line breaks to the cue text
      if (cueText.length > 0) {
        // Split the text into smaller chunks of around 30-40 characters
        // Try to split at natural break points like punctuation
        const formattedText = formatText(cueText);
        modifiedLines.push(formattedText);
        cueText = '';
      }
      
      modifiedLines.push('');
      continue;
    }
    
    // If we're in a cue, collect the text
    if (inCue) {
      cueText += line + ' ';
      continue;
    }
    
    // Not in a cue, just add the line as is
    modifiedLines.push(line);
  }
  
  return modifiedLines.join('\n');
}

// Helper function to format text with line breaks
function formatText(text: string): string {
  text = text.trim();
  
  // If text is short, return as is
  if (text.length < 40) {
    return text;
  }
  
  // Try to split at natural break points
  const breakPoints = ['. ', '! ', '? ', ': ', '; '];
  for (const bp of breakPoints) {
    if (text.includes(bp)) {
      return text.split(bp).join(bp + '\n');
    }
  }
  
  // If no natural break points, split after approximately 40 characters
  // Try to split at a space to avoid breaking words
  const words = text.split(' ');
  let result = '';
  let lineLength = 0;
  
  for (const word of words) {
    if (lineLength + word.length > 40) {
      result += '\n' + word + ' ';
      lineLength = word.length + 1;
    } else {
      result += word + ' ';
      lineLength += word.length + 1;
    }
  }
  
  return result.trim();
}

// Function to convert VTT file to data URI
function convertVttFileToDataURI(filePath: string): string {
  // Read the VTT file
  let vttContent: string = fs.readFileSync(filePath, 'utf8');
  
  // Add line breaks to the VTT content
  vttContent = addLineBreaksToVTT(vttContent);
  
  // Write the modified VTT file back
  fs.writeFileSync(filePath, vttContent);
  
  // Convert VTT content to Base64
  const base64Content: string = Buffer.from(vttContent).toString('base64');
  
  // Create data URI
  return `data:text/vtt;base64,${base64Content}`;
}

export async function generateSubtitles(audioFilePath: string): Promise<string | undefined> {
  try {
    // Get the base name of the audio file (without extension)
    const baseFileName: string = path.basename(audioFilePath, path.extname(audioFilePath));
    
    // Run Whisper to generate the VTT file
    await runWhisperCommand(audioFilePath, outputDir);
    
    // Path to the generated VTT file
    const vttPath: string = path.join(outputDir, `${baseFileName}.vtt`);
    
    console.log('Checking for VTT file at:', vttPath);
    
    // Check if VTT file was generated
    if (!fs.existsSync(vttPath)) {
      throw new Error('VTT file was not generated');
    }
    
    console.log('VTT file found, converting to data URI...');
    
    // Convert VTT file to data URI
    const dataURI: string = convertVttFileToDataURI(vttPath);
    
    // Log the data URI
    console.log(dataURI);
    
    // Optionally save to file
    const outputPath: string = path.join(__dirname, 'subtitles-data-uri.txt');
    fs.writeFileSync(outputPath, dataURI);
    
    console.log(`Data URI saved to: ${outputPath}`);
    
    return dataURI;
  } catch (error) {
    console.error('Error generating subtitles:', error);
    return undefined;
  }
}
