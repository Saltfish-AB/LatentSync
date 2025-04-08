import { ModelQueueJob } from "./models";
import {
  getAllDocuments,
  getDocumentById,
  getDocumentByPath,
  updateDocument,
  updateDocumentByPath,
} from "./helpers/firestore";
import { textToSpeech } from "./helpers/eleven-labs";
import path from 'path';
import { uploadFileToGCS } from "./helpers/gcs";
import { FieldValue, Timestamp } from "firebase-admin/firestore";
import { concatVideos } from "./helpers/ffmpeg";
import { Agent, Dispatcher, setGlobalDispatcher } from 'undici';
import { downloadFile } from "./helpers/download";
import { generateSubtitles } from "./helpers/whisper";
import { removeFile } from "./helpers/file";

const dispatcher = new Agent({
  headersTimeout: 0, // 0 = no timeout
  connectTimeout: 0, // 0 = no timeout
  bodyTimeout: 0     // 0 = no timeout
});

setGlobalDispatcher(dispatcher);

const updateStatus = async (
  job: ModelQueueJob,
  status: string,
  outputUrl?: string,
  gifUrl?: string
) => {
  const updateData: { status: string; outputUrl?: string; gifUrl?: string; } = { status };
  if (outputUrl) {
    updateData.outputUrl = outputUrl;
  }
  if(gifUrl){
    updateData.gifUrl = gifUrl;
  }
  await updateDocument("latent-sync-jobs", job.id, updateData);
  if (job.clipId) {
    try {
      await updateDocument("clips", job.clipId, updateData);
    } catch (error) {
      console.error(`Failed to update clip ${job.clipId}:`, error);
    }
  }
  if(job.params?.dynamicClipChildId) {
    await updateDocument(`dynamic-clips/${job.params.dynamicClipId}/start-segments`, job.params.dynamicClipChildId, updateData);
    if(status === "completed") {
      await updateDocument("dynamic-clips", job.params.dynamicClipId, {
        completedChildren: FieldValue.increment(1)
      });
      const dynamicClip = await getDocumentById("dynamic-clips", job.params.dynamicClipId);
      if(!dynamicClip) {
        return;
      }
      console.log(dynamicClip)
      if(dynamicClip.completedChildren === dynamicClip.totalChildren){
        await updateDocument("dynamic-clips", job.params.dynamicClipId, {
          status: "completed"
        });
      }
    }
  } else if(job.params?.dynamicClipId){
    try {
      if(status === "completed") {
        const dynamicClip = await getDocumentById("dynamic-clips", job.params.dynamicClipId)
        let data: { outputUrl: string | undefined; status?: string } = { outputUrl };
        if(dynamicClip!.totalChildren === 0){
          data.status = "completed"
        }
        await updateDocument("dynamic-clips", job.params.dynamicClipId, data);
      } else {
        await updateDocument("dynamic-clips", job.params.dynamicClipId, updateData);
      }
    } catch (error) {
      console.error(`Failed to update clip ${job.clipId}:`, error);
    }
  } else if(job?.dynamicInserts){
    try {
      const dynamicInserts = job.dynamicInserts;
      if(status === "completed") {
        await updateDocument("dynamic-inserts", dynamicInserts.dynamicInsertId, {
          completedChildren: FieldValue.increment(1)
        });
        const dynamicInsert = await getDocumentById("dynamic-inserts", dynamicInserts.dynamicInsertId)
        console.log(dynamicInsert)
        console.log(dynamicInsert!.completedChildren, dynamicInsert!.totalChildren, dynamicInsert!.completedChildren >= dynamicInsert!.totalChildren)
        if(dynamicInsert!.completedChildren >= dynamicInsert!.totalChildren){
          await updateDocument("dynamic-inserts", dynamicInserts.dynamicInsertId, {
            status: "completed"
          });
        }
      } else {
        await updateDocument(`dynamic-inserts`, dynamicInserts.dynamicInsertId, updateData);
      }
      await updateDocument(`dynamic-inserts/${dynamicInserts.dynamicInsertId}/inserts`, dynamicInserts.insertId, updateData);
    } catch (error) {
      console.error(`Failed to update clip ${job.clipId}:`, error);
    }
  }
  if(job.params?.clipId){
    await updateDocument(`clips`, job.params.clipId, updateData);
  }
};

const runLoop = async () => {
    while (true) {
      const pendingJobs = await getAllDocuments(
        "latent-sync-jobs",
        [["status", "==", "pending"]],
        "created_at",
        "asc"
      );
      
      const waitingJobs = await getAllDocuments(
        "latent-sync-jobs",
        [["status", "==", "waiting-dependency"]],
        "created_at",
        "asc"
      );

      if (pendingJobs.length === 0) {
        console.log("No jobs found. Waiting for more jobs...");
      }

      for(const job of pendingJobs){
        await handleJob(job as ModelQueueJob);
      }

      for (const job of waitingJobs) {
        console.log(`Processing waiting job with ID: ${job.id}`);
        
        try {
          console.log(`Fetching dependency document: ${job.dependencyDoc}`);
          const jobData = await getDocumentByPath(job.dependencyDoc);
          
          console.log(`Dependency document status: ${jobData?.status || 'undefined'}`);
          
          if (!jobData) {
            console.warn(`Dependency document not found: ${job.dependencyDoc}`);
            continue;
          }
          
          if (jobData.status === "completed") {
            console.log(`Updating dependency document ${job.dependencyDoc} to 'pending' status`);
            
            await updateDocument("latent-sync-jobs", job.id, {
              status: "pending"
            });
            
            console.log(`Successfully updated dependency document ${job.dependencyDoc}`);
          } else {
            console.log(`Skipping update - dependency document status is not 'completed'`);
          }
        } catch (error) {
          console.error(`Error processing job ${job.id} with dependency ${job.dependencyDoc}:`, error);
          // Optionally: You might want to handle this error specifically, 
          // e.g., updating the job's status to "error"
        }
      }
      
      await new Promise((resolve) => setTimeout(resolve, 5000));
  }
};

const handleJob = async (job: ModelQueueJob) => {
  await updateStatus(job, "running");
  let generatedAudioUrl;
  let nextText = "";
  if(job.params?.elevenLabsVoiceId){
    const outputFilePath = path.resolve(__dirname, 'output.mp3');
    nextText = job.params.nextText || "";
    await textToSpeech(job.params.elevenLabsVoiceId, job.params.textPrompt, outputFilePath, nextText)
    await uploadFileToGCS(outputFilePath, `elevenlabs/${job.id}.mp3`, "audio/mpeg")
    generatedAudioUrl = `https://storage.saltfish.ai/elevenlabs/${job.id}.mp3`;
  }
  console.log(job)
  const audioUrl = job.params.audioUrl || generatedAudioUrl;
  const isDynamicClip = job.params.isDynamicClip || false;

  const avatar = await getDocumentById("avatar-videos", job.params.avatarVideoId)

  try {
    const url = 'http://localhost:8000/process';
    const payload = {
      id: job.id,
      video_id: job.params.avatarVideoId,
      audio_url: audioUrl,
      start_from_backwards: job.params.dynamicClipChildId ? true : false,
      is_dynamic_clip: isDynamicClip,
      text: `${job.params.textPrompt} ${nextText}`,
      use_darken: avatar?.darkenData?.needs_correction ?? false,
      brightness_factor: avatar?.darkenData?.recommended_brightness_factor ?? undefined,
    };

    console.log(payload)

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const responseText = await response.text();
        console.log('Response text:', responseText);
        throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const result = await response.json();
    console.log(result)
    
    if(!job.params.dynamicClipId && !job.params.dynamicClipChildId){
      await updateStatus(job, "completed", result.output_url);
    } else if(job.params.dynamicClipChildId) {
      const dynamicClip = await getDocumentById("dynamic-clips", job.params.dynamicClipId);
      if(!dynamicClip) {
        return;
      }
      const outputFilePath = path.resolve(__dirname, 'output.mp4');
      await concatVideos(result.output_url, dynamicClip.outputUrl, outputFilePath)
      await uploadFileToGCS(outputFilePath, `dynamic-clips/${job.params.dynamicClipId}/${job.params.dynamicClipChildId}.mp4`, "video/mp4")
      const gifPath = `${job.params.dynamicClipChildId}.gif`
      await downloadFile(result.gif_url, gifPath)
      await uploadFileToGCS(gifPath, `gifs/${job.params.dynamicClipId}/${job.params.dynamicClipChildId}.gif`, "image/gif")
      await removeFile(gifPath)

      if(isDynamicClip && job.params.dynamicClipId && job.params.dynamicClipChildId){
        const dataUri = await generateSubtitles(outputFilePath)
        await updateDocument(`dynamic-clips/${job.params.dynamicClipId}/start-segments`, job.params.dynamicClipChildId, {
          subtitlesData: dataUri
        });
      }

      await updateStatus(job, "completed", `https://storage.saltfish.ai/dynamic-clips/${job.params.dynamicClipId}/${job.params.dynamicClipChildId}.mp4`, `https://storage.saltfish.ai/gifs/${job.params.dynamicClipId}/${job.params.dynamicClipChildId}.gif`);
    } else if(job.params.dynamicClipId) {
      const startSegments = await getAllDocuments(`dynamic-clips/${job.params.dynamicClipId}/start-segments`);
      for(const startSegment of startSegments){
        await updateDocument("latent-sync-jobs", startSegment.jobId, {status: "pending"})
      }
      await updateStatus(job, "completed", result.output_url);
    }
  } catch (error) {
    console.error(`Failed to handle job ${job.id}:`, error);
    await updateStatus(job, "failed");
    return;
  }

  
};

runLoop();
